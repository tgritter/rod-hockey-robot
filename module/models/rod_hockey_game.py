import asyncio
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from ..constants import PLAYERS

from typing_extensions import Self
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import *
from viam.utils import ValueTypes

class RodHockeyGame(Generic, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-rod-hockey", "rod-hockey-game"), "rod_hockey_game"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Generic service.
        The default implementation sets the name from the `config` parameter.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        self = super().new(config, dependencies)
        return self

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        return PLAYERS, []

    # Background task running the autonomous play loop, or None when auto-play is off.
    _auto_task = None
    # Flag the loop checks each cycle; setting it False stops auto-play gracefully.
    _auto_run = False

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        # The web UI only ever sends one of three actions. Dispatch on it.
        action = command.get("action")

        # Home every rod (run once after a robot restart, before any play works).
        if action == "home":
            from home import _PLAYER_TO_GANTRY  # reuse the existing player->gantry map
            from robot.execution import _get_robot
            from viam.components.gantry import Gantry
            robot = await _get_robot()
            for name in _PLAYER_TO_GANTRY.values():
                await Gantry.from_robot(robot, name).home()
            return {"message": "All rods homed."}

        # Report puck detection + which side it's on, plus current auto-play state.
        if action == "status":
            from robot.vision import get_puck_field_coordinates
            u, _ = await get_puck_field_coordinates()
            return {
                "auto_play": self._auto_run,
                "puck_detected": u is not None,
                "puck_side": ("left" if u < 0.5 else "right") if u is not None else "",
            }

        # Start or stop the autonomous play loop (graceful stop via the flag).
        if action == "auto_play":
            if command.get("enabled"):
                # Start the loop once; the flag keeps it alive until we clear it.
                if self._auto_task is None or self._auto_task.done():
                    self._auto_run = True
                    from main import run_loop
                    self._auto_task = asyncio.create_task(
                        run_loop(should_continue=lambda: self._auto_run)
                    )
            else:
                # Loop finishes its current play, then exits at the next cycle check.
                self._auto_run = False
            return {"auto_play": self._auto_run}

        raise ValueError(f"unknown action: {action}")

    async def get_status(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`get_status` is not implemented")
        raise NotImplementedError()
