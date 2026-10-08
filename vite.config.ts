import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Relative base so the build works under any path, e.g. GitHub Pages' /<repo>/.
export default defineConfig({ base: './', plugins: [react(), tailwindcss()] });
