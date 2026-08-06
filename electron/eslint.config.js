const tseslint = require("typescript-eslint");

module.exports = tseslint.config(
  {
    ignores: ["dist/**", "node_modules/**", "renderer/**", "eslint.config.js"],
  },
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.ts", "tests/**/*.ts"],
    languageOptions: {
      parserOptions: {
        // Emit tsconfig is src-only; this project includes tests for typed lint.
        project: "./tsconfig.eslint.json",
        tsconfigRootDir: __dirname,
      },
    },
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" },
      ],
    },
  },
);
