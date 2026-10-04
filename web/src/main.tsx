import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import Layout from "./components/Layout";
import "./index.css";
import Agent from "./pages/Agent";
import Backtest from "./pages/Backtest";
import Dashboard from "./pages/Dashboard";
import DataPage from "./pages/DataPage";
import ModelLab from "./pages/ModelLab";
import NotFound from "./pages/NotFound";
import Screener from "./pages/Screener";
import Stock from "./pages/Stock";

const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 5 * 60_000, retry: 1, refetchOnWindowFocus: false } } });

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: "screener", element: <Screener /> },
      { path: "stock/:ticker", element: <Stock /> },
      { path: "agent", element: <Agent /> },
      { path: "backtest", element: <Backtest /> },
      { path: "model", element: <ModelLab /> },
      { path: "data", element: <DataPage /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
