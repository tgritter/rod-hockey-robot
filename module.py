import asyncio
from viam.module.module import Module
from module.models.rod_hockey_game import RodHockeyGame as RodHockeyGameModel


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
