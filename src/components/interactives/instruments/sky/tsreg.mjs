// `node --import ./tsreg.mjs …`: registers tsresolve.mjs (extensionless .ts imports) for sweeps.
import { register } from 'node:module';
register('./tsresolve.mjs', import.meta.url);
