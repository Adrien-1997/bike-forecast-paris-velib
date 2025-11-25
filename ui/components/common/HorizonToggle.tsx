// components/common/HorizonToggle.tsx
//
// =============================================================================
// Sélecteur d’horizon (15 min / 60 min par défaut).
//
// Rôle :
// - Afficher un switch compact pour choisir l’horizon de prévision (ex: 15 ou 60 minutes).
// - Gérer l’accessibilité au clavier (Enter / Espace / flèches gauche/droite).
// - Rester agnostique du contexte : le parent fournit `value` et `onChange`.
//
// UX :
// - Visuellement, un petit switch avec labels texte “15 min” / “60 min” de part et d’autre.
// - Comportement : clic → bascule, flèches → choix explicite du côté gauche/droit.
// =============================================================================

import { useCallback } from "react";
import type { KeyboardEvent } from "react";

type Props = {
  /** Valeur courante de l’horizon, en minutes (ex: 15 ou 60). */
  value?: number;             // 15 ou 60 (défaut : 60)
  /** Callback déclenché à chaque changement d’horizon. */
  onChange: (v: number) => void;
  /** Valeur numérique côté gauche du switch (défaut 15). */
  leftValue?: number;         // défaut 15
  /** Valeur numérique côté droit du switch (défaut 60). */
  rightValue?: number;        // défaut 60
  /** Label texte affiché à gauche (défaut "15 min"). */
  leftLabel?: string;         // défaut "15 min"
  /** Label texte affiché à droite (défaut "60 min"). */
  rightLabel?: string;        // défaut "60 min"
  /** Libellé ARIA du switch (lecteurs d’écran). */
  ariaLabel?: string;
  /** Variante plus compacte du switch (padding / hauteur réduits). */
  dense?: boolean;
};

export default function HorizonToggle({
  value = 60,                 // ← 60 min par défaut (correspond à rightValue par défaut)
  onChange,
  leftValue = 15,
  rightValue = 60,
  leftLabel = "15 min",
  rightLabel = "60 min",
  ariaLabel = "Horizon",
  dense = true,
}: Props) {
  // `checked` indique si le switch est visuellement sur la "droite"
  // (c’est-à-dire sur `rightValue`).
  const checked = value === rightValue;

  /**
   * Basculement simple : si on est sur la droite → gauche, sinon → droite.
   * Utilisé à la fois pour le clic souris et certaines interactions clavier.
   */
  const toggle = useCallback(() => {
    onChange(checked ? leftValue : rightValue);
  }, [checked, leftValue, rightValue, onChange]);

  /**
   * Gestion clavier (accessibilité) :
   * - Espace / Enter → toggle,
   * - Flèche gauche  → force `leftValue`,
   * - Flèche droite  → force `rightValue`.
   */
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
      {/* Label fixe "Horizon" devant le switch (pure déco, pas ARIA). */}
      <span className="hz-switch__label">Horizon :</span>

      {/* Label texte côté gauche (ex: "15 min"), masqué aux lecteurs d’écran. */}
      <span className="hz-switch__text" aria-hidden="true">{leftLabel}</span>

      {/* Bouton principal du switch.
          - role="switch" + aria-checked : composant accessible.
          - aria-label : texte lu par les lecteurs d’écran. */}
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={ariaLabel}
        className={["hz-switch__control", dense ? "is-dense" : ""].join(" ")}
        onClick={toggle}
        onKeyDown={onKeyDown}
      >
        {/* Thumb visuel du switch (position gérée en CSS via aria-checked). */}
        <span className="hz-switch__thumb" />
      </button>

      {/* Label texte côté droit (ex: "60 min"), masqué aux lecteurs d’écran. */}
      <span className="hz-switch__text" aria-hidden="true">{rightLabel}</span>
    </div>
  );
}
