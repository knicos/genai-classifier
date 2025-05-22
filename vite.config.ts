/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import dts from 'vite-plugin-dts';
import { libInjectCss } from 'vite-plugin-lib-inject-css';
import { extname, relative, resolve } from 'path';
import { glob } from 'glob';
import { fileURLToPath } from 'url';

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        react(),
        libInjectCss(),
        dts({
            tsconfigPath: './tsconfig.build.json',
            include: ['lib'],
            rollupTypes: false,
            beforeWriteFile: (filePath, content) => ({
                filePath: filePath.replace('lib/', ''),
                content,
            }),
        }),
    ],
    test: {
        environment: 'jsdom',
        setupFiles: './src/setupTests.ts',
        clearMocks: true,
        coverage: {
            provider: 'v8',
            reporter: ['cobertura', 'html'],
            include: ['lib/**/*.{ts,tsx}'],
        },
    },
    resolve: {
        alias: {
            '@public': resolve(__dirname, './public'),
            '@base': resolve(__dirname, './lib'),
        },
    },
    build: {
        copyPublicDir: true,
        rollupOptions: {
            external: [
                'react',
                'react-dom',
                'react/jsx-runtime',
                '@mui/material',
                '@mui/icons-material',
                '@emotion/react',
                '@emotion/styled',
                'react-i18next',
                'recoil',
            ],
            output: {
                assetFileNames: 'assets/[name][extname]',
                entryFileNames: '[name].js',
            },
            input: Object.fromEntries(
                glob
                    .sync('lib/**/*.{ts,tsx}', {
                        ignore: ['lib/**/*.d.ts', 'lib/**/*.test.ts', 'lib/**/*.test.tsx'],
                    })
                    .map((file) => [
                        // The name of the entry point
                        // lib/nested/foo.ts becomes nested/foo
                        relative('lib', file.slice(0, file.length - extname(file).length)),
                        // The absolute path to the entry file
                        // lib/nested/foo.ts becomes /project/lib/nested/foo.ts
                        fileURLToPath(new URL(file, import.meta.url)),
                    ])
            ),
        },
        lib: {
            entry: resolve(__dirname, 'lib/main.tsx'),
            formats: ['es'],
        },
    },
});
