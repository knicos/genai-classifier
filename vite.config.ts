/// <reference types="vitest" />
import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
import { libInjectCss } from 'vite-plugin-lib-inject-css';
import { extname, relative, resolve } from 'path';
import { glob } from 'glob';
import { fileURLToPath } from 'url';

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
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
            include: ['lib/**/*.ts'],
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
            output: {
                assetFileNames: 'assets/[name][extname]',
                entryFileNames: '[name].js',
            },
            input: Object.fromEntries(
                glob
                    .sync('lib/**/*.ts', {
                        ignore: ['lib/**/*.d.ts', 'lib/**/*.test.ts'],
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
            entry: resolve(__dirname, 'lib/main.ts'),
            formats: ['es'],
        },
    },
});
