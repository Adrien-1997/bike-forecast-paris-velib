// ui/components/monitoring/LoadingBar.tsx

// =============================================================================
// Barre de chargement globale pour les pages Monitoring.
//
// Rôle :
// - Afficher un état de chargement global en haut de la page (ou dans une zone dédiée) :
//     • "loading"  : chargement en cours,
//     • "success"  : chargement réussi (flash puis disparition),
//     • "error"    : erreur de chargement (API ou hors-ligne),
//     • "idle"     : pas de barre (composant non rendu).
// - Adapter le message d’erreur si l’utilisateur est hors ligne.
// - Proposer un bouton "Réessayer" si une callback `onRetry` est fournie.
//
// Accessibilité :
// - `role="status"` + `aria-live` pour que les lecteurs d’écran annoncent
//   les changements (loading, success, error).
// - `aria-label` adapté en fonction de `status`.
// =============================================================================

import { useEffect, useMemo, useState } from "react";

export type LoadingBarStatus = "idle" | "loading" | "success" | "error";

export default function LoadingBar({
  status,
  successFlashMs = 500,
  onFlashEnd,
  onRetry,                 // optionnel : action de retry
  errorLabel,              // optionnel : message d’erreur personnalisé
}: {
  /** État courant de la barre (idle, loading, success, error). */
  status: LoadingBarStatus;
  /**
   * Durée (en millisecondes) du flash "success" :
   * - au bout de ce délai, la barre se masque automatiquement et
   *   déclenche `onFlashEnd` si défini.
   */
  successFlashMs?: number;
  /**
   * Callback appelée à la fin du flash "success", après masquage de la barre.
   * Permet par exemple au parent de repasser en `idle`.
   */
  onFlashEnd?: () => void;
  /**
   * Callback appelée quand l’utilisateur clique sur le bouton "Réessayer".
   * Si non fournie, le bouton n’est pas affiché.
   */
  onRetry?: () => void;
  /**
   * Message d’erreur personnalisé. Si non fourni, on construit un message
   * par défaut en fonction de l’état réseau (offline / erreur API).
   */
  errorLabel?: string;
}) {
  const [show, setShow] = useState(false);

  // Détection réseau pour un message plus utile :
  // - si navigator.onLine === false → l’utilisateur est hors ligne,
  // - sinon, on considère que l’erreur vient plutôt de l’API ou du réseau externe.
  const isOffline = useMemo(() => {
    if (typeof navigator === "undefined" || typeof navigator.onLine !== "boolean") return false;
    return !navigator.onLine;
  }, [status]);

  useEffect(() => {
    let t: any;

    if (status === "loading" || status === "error") {
      // Loading ou erreur → la barre doit être visible
      setShow(true);
    } else if (status === "success") {
      // En cas de succès :
      // - on affiche la barre,
      // - puis on la masque après `successFlashMs` et on prévient le parent.
      setShow(true);
      t = setTimeout(() => {
        setShow(false);
        onFlashEnd?.();
      }, successFlashMs);
    } else {
      // "idle" → la barre est cachée
      setShow(false);
    }

    return () => clearTimeout(t);
  }, [status, successFlashMs, onFlashEnd]);

  // Rien à afficher si la barre ne doit pas être visible
  if (!show) return null;

  const role = "status";
  const ariaLabel =
    status === "loading"
      ? "Chargement en cours"
      : status === "success"
      ? "Chargement réussi"
      : "Erreur de chargement";

  // Libellé d’erreur (offline vs API), surchargeable via `errorLabel`
  const errText =
    errorLabel ??
    (isOffline
      ? "Hors ligne — vérifiez votre connexion."
      : "Erreur API — réessayez.");

  return (
    <div
      className={`loadingbar ${
        status === "loading" ? "is-loading" : status === "success" ? "is-success" : "is-error"
      }`}
      role={role}
      aria-live={status === "error" ? "assertive" : "polite"}
      aria-label={ariaLabel}
    >
      {/* Piste + barre animée (gérée en CSS via .is-loading / .is-success / .is-error) */}
      <div className="loadingbar__track">
        <div className="loadingbar__bar" />
      </div>

      {/* En cas d’erreur, on affiche le message + éventuel bouton "Réessayer" */}
      {status === "error" && (
        <div className="loadingbar__msg">
          <span className="loadingbar__text">{errText}</span>
          {onRetry && (
            <button
              type="button"
              className="loadingbar__retry"
              onClick={onRetry}
              aria-label="Réessayer"
            >
              Réessayer
            </button>
          )}
        </div>
      )}
    </div>
  );
}
