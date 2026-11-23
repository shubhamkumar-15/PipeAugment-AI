"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setMounted(true);
      const stored = localStorage.getItem("pipeaugment_theme");
      if (stored === "dark") {
        setTheme("dark");
      } else {
        setTheme("light");
      }
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    localStorage.setItem("pipeaugment_theme", nextTheme);

    const root = document.documentElement;
    if (nextTheme === "dark") {
      root.setAttribute("data-theme", "dark");
    } else {
      root.removeAttribute("data-theme");
    }
  };

  if (!mounted) {
    return (
      <div className="w-8 h-8 rounded-lg border border-[var(--border)] bg-[var(--bg-2)]" />
    );
  }

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="w-8 h-8 rounded-lg border border-[var(--border)] bg-[var(--bg)] hover:bg-[var(--bg-2)] text-[var(--muted)] hover:text-[var(--ink)] transition-colors flex items-center justify-center cursor-pointer shadow-xs"
      title={`Switch to ${theme === "light" ? "Dark" : "Light"} mode`}
      aria-label="Toggle visual theme"
    >
      {theme === "light" ? (
        <Moon className="w-4 h-4 text-[var(--ink)]" />
      ) : (
        <Sun className="w-4 h-4 text-[var(--gold)]" />
      )}
    </button>
  );
}
