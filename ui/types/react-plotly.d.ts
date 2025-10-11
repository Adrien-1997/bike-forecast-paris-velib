// ui/types/react-plotly.d.ts
declare module "react-plotly.js" {
  import { ComponentType, CSSProperties } from "react";
  import type { Layout, Config, Data } from "plotly.js";

  export interface PlotProps {
    data: Partial<Data>[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    style?: CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: { data: Partial<Data>[]; layout: Partial<Layout> }) => void;
    onUpdate?: (figure: { data: Partial<Data>[]; layout: Partial<Layout> }) => void;
  }

  const Plot: ComponentType<PlotProps>;
  export default Plot;
}