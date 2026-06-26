// Reads the credentials Viam injects as cookies when a user logs in to the app.
// Trimmed from the Viam CLI app scaffold to just the single-machine path.
// Import points at a CDN so the page needs no build step.
import { getCookie } from 'https://esm.sh/typescript-cookie@1.0.6';

export function getHostAndCredentials() {
    // Direct host + api-key cookies (the common single-machine case).
    const host = getCookie('host');
    const apiKeyId = getCookie('api-key-id');
    const apiKeySecret = getCookie('api-key');
    if (host && apiKeyId && apiKeySecret) {
        return {
            host,
            credentials: { type: 'api-key', payload: apiKeySecret, authEntity: apiKeyId },
        };
    }

    // Fallback: per-machine cookie keyed by the /machine/<id> path Viam serves under.
    const parts = window.location.pathname.split('/');
    if (parts.length >= 3 && parts[1] === 'machine') {
        const cookieData = getCookie(parts[2]);
        if (cookieData) {
            const parsed = JSON.parse(cookieData);
            const id = parsed?.apiKey?.id;
            const key = parsed?.apiKey?.key;
            const h = parsed?.hostname;
            if (h && id && key) {
                return { host: h, credentials: { type: 'api-key', payload: key, authEntity: id } };
            }
        }
    }

    // No credentials found (e.g. opened outside the Viam app shell).
    return { host: '', credentials: { type: 'api-key', payload: '', authEntity: '' } };
}
