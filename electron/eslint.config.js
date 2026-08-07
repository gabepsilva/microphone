const tseslint = require("typescript-eslint");

module.exports = tseslint.config(
  {
    ignores: [
      "dist/**",
      "node_modules/**",
      "coverage/**",
      "renderer/**",
      "eslint.config.js",
      // Planted CJS shapes for sandbox-preload gates (intentionally use require).
      "tests/fixtures/**/*.js",
    ],
  },
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.ts", "tests/**/*.ts", "tools/**/*.ts"],
    languageOptions: {
      parserOptions: {
        // Emit tsconfig is src-only; check project includes tests for typed lint.
        project: "./tsconfig.check.json",
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
