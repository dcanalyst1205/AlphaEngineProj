import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "var(--background)",
                foreground: "var(--foreground)",
                premium: {
                    50: "#f5f7fa",
                    100: "#e4e9f2",
                    200: "#ccd5e6",
                    300: "#a9b9d4",
                    400: "#7e96bc",
                    500: "#5d78a3",
                    600: "#485f88",
                    700: "#3d4d6e",
                    800: "#35415c",
                    900: "#2f394f",
                    950: "#1f2535",
                }
            },
        },
    },
    plugins: [],
};
export default config;
