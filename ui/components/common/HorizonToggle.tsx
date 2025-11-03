// components/common/HorizonToggle.tsx
import { useCallback } from "react";
import type { KeyboardEvent } from "react";

type Props = {
  value?: number;             // 15 ou 60 (défaut : 60)
  onChange: (v: number) => void;
  leftValue?: number;         // défaut 15
  rightValue?: number;        // défaut 60
  leftLabel?: string;         // défaut "15 min"
  rightLabel?: string;        // défaut "60 min"
  ariaLabel?: string;
  dense?: boolean;
};

export default function HorizonToggle({
  value = 60,                 // ← 60 min par défaut
  onChange,
  leftValue = 15,
  rightValue = 60,
  leftLabel = "15 min",
  rightLabel = "60 min",
  ariaLabel = "Horizon",
  dense = true,
}: Props) {
  const checked = value === rightValue;

  const toggle = useCallback(() => {
    onChange(checked ? leftValue : rightValue);
  }, [checked, leftValue, rightValue, onChange]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLButtonElement>) => {
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        toggle();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        onChange(leftValue);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onChange(rightValue);
      }
    },
    [toggle, onChange, leftValue, rightValue]
  );

  return (
    <div className="monitoring hz-switch">
      <span className="hz-switch__label">Horizon :</span>

      <span className="hz-switch__text" aria-hidden="true">{leftLabel}</span>

      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={ariaLabel}
        className={["hz-switch__control", dense ? "is-dense" : ""].join(" ")}
        onClick={toggle}
        onKeyDown={onKeyDown}
      >
        <span className="hz-switch__thumb" />
      </button>

      <span className="hz-switch__text" aria-hidden="true">{rightLabel}</span>
    </div>
  );
}
