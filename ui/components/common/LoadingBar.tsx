// ui/components/monitoring/LoadingBar.tsx
import { useEffect, useMemo, useState } from "react";

export type LoadingBarStatus = "idle" | "loading" | "success" | "error";

export default function LoadingBar({
  status,
  successFlashMs = 500,
  onFlashEnd,
  onRetry,                 // optionnel : action de retry
  errorLabel,              // optionnel : message d’erreur personnalisé
}: {
  status: LoadingBarStatus;
  successFlashMs?: number;
  onFlashEnd?: () => void;
  onRetry?: () => void;
  errorLabel?: string;
}) {
  const [show, setShow] = useState(false);

  // Détection réseau pour un message plus utile
  const isOffline = useMemo(() => {
    if (typeof navigator === "undefined" || typeof navigator.onLine !== "boolean") return false;
    return !navigator.onLine;
  }, [status]);

  useEffect(() => {
    let t: any;
    if (status === "loading" || status === "error") {
      setShow(true);
    } else if (status === "success") {
      setShow(true);
      t = setTimeout(() => {
        setShow(false);
        onFlashEnd?.();
      }, successFlashMs);
    } else {
      setShow(false);
    }
    return () => clearTimeout(t);
  }, [status, successFlashMs, onFlashEnd]);

  if (!show) return null;

  const role = "status";
  const ariaLabel =
    status === "loading"
      ? "Chargement en cours"
      : status === "success"
      ? "Chargement réussi"
      : "Erreur de chargement";

  // Libellé d’erreur (offline vs API), surchargeable via props
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
      <div className="loadingbar__track">
        <div className="loadingbar__bar" />
      </div>

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
