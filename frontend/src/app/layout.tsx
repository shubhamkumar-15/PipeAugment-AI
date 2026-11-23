import type { Metadata } from "next";
import { Bebas_Neue, DM_Sans, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import { SplashPreloader, MagneticCursor } from "@/components/motion";
import "./globals.css";

const bebasNeue = Bebas_Neue({
  variable: "--font-bebas-neue",
  subsets: ["latin"],
  weight: "400",
  display: "swap",
});

const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "PipeAugment AI — Condition-Controlled Pipeline Augmentation",
  description:
    "Industrial visual inspection data augmentation & synthetic condition generation engine preserving corrosion features.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${bebasNeue.variable} ${dmSans.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  const stored = localStorage.getItem('pipeaugment_theme');
                  if (stored === 'dark') {
                    document.documentElement.setAttribute('data-theme', 'dark');
                  } else {
                    document.documentElement.removeAttribute('data-theme');
                  }
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-[var(--bg)] text-[var(--ink-2)] font-body antialiased selection:bg-[var(--saffron)] selection:text-white relative">
        <SplashPreloader />
        <MagneticCursor />
        {children}
      </body>
    </html>
  );
}
