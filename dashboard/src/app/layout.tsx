import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "Alpha Engine | Performance Dashboard",
    description: "ML-Driven Quantitative Alpha Generation",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className="antialiased font-sans">{children}</body>
        </html>
    );
}
