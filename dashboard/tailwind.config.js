/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                premium: {
                    400: "#5d78a3",
                    500: "#4a638c",
                    600: "#3a4e6e",
                    700: "#2a394f",
                    800: "#1a2432",
                    900: "#2f394f",
                    950: "#1f2535",
                }
            },
            animation: {
                "pulse-glow": "pulse-glow 4s ease-in-out infinite",
                "data-stream": "data-stream 20s linear infinite",
            },
            keyframes: {
                "pulse-glow": {
                    "0%, 100%": { opacity: "0.05", transform: "scale(1)" },
                    "50%": { opacity: "0.15", transform: "scale(1.05)" },
                },
                "data-stream": {
                    "0%": { transform: "translateY(0)" },
                    "100%": { transform: "translateY(-50%)" },
                },
            },
        },
    },
    plugins: [],
}
