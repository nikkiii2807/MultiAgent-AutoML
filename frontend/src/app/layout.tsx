import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AutoML Studio",
  description: "Notebook-style AutoML workspace with an AI copilot and interactive analytics outputs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
