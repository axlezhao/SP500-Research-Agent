import { useEffect, useState, useSyncExternalStore } from "react";

export type ThemeChoice = "system" | "light" | "dark";

function readChoice(): ThemeChoice {
  try {
    const stored = localStorage.getItem("theme");
    return stored === "light" || stored === "dark" ? stored : "system";
  } catch {
    return "system";
  }
}

export function useThemeChoice(): [ThemeChoice, (c: ThemeChoice) => void] {
  const [choice, setChoice] = useState<ThemeChoice>(readChoice);
  useEffect(() => {
    const root = document.documentElement;
    if (choice === "system") delete root.dataset.theme;
    else root.dataset.theme = choice;
    try {
      if (choice === "system") localStorage.removeItem("theme");
      else localStorage.setItem("theme", choice);
    } catch {
      /* storage unavailable: the choice still applies for this visit */
    }
    window.dispatchEvent(new Event("themechange"));
  }, [choice]);
  return [choice, setChoice];
}

function subscribe(callback: () => void) {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", callback);
  window.addEventListener("themechange", callback);
  return () => {
    media.removeEventListener("change", callback);
    window.removeEventListener("themechange", callback);
  };
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Resolved colours for charts; re-renders when the theme changes. */
export function useChartColors() {
  const key = useSyncExternalStore(subscribe, () => `${document.documentElement.dataset.theme ?? ""}:${window.matchMedia("(prefers-color-scheme: dark)").matches}`);
  const [colors, setColors] = useState(() => readColors());
  useEffect(() => {
    setColors(readColors());
  }, [key]);
  return colors;
}

function readColors() {
  return {
    series: [css("--series-1"), css("--series-2"), css("--series-3")],
    accent: css("--accent"),
    grid: css("--grid"),
    border: css("--border"),
    text: css("--text"),
    text2: css("--text-2"),
    muted: css("--muted"),
    surface: css("--surface"),
    pos: css("--pos"),
    neg: css("--neg"),
  };
}
