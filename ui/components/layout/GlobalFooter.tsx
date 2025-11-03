import Link from "next/link";
import { ReactNode } from "react";

type SocialItem = {
  href: string;
  label: string;           // ex: "LinkedIn Adrien Morel"
  icon?: ReactNode;        // facultatif : <svg/>
};

export default function GlobalFooter({
  siteName = "velo-paris.fr",
  projectNote = "Projet indépendant",
  year = new Date().getFullYear(),
  social = [
    {
      href: "https://www.linkedin.com/in/adrien-morel/",
      label: "Suivez moi sur LinkedIn",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" aria-hidden="true" focusable="false">
          <path
            fill="currentColor"
            d="M100.3 448H7.4V149.1h92.9V448zM53.8 108.1C24.1 108.1 0 83.9 0 54.2 0 24.5 24.1.4 53.8.4s53.8 24.1 53.8 53.8c0 29.7-24.1 53.9-53.8 53.9zM447.9 448h-92.7V302.4c0-34.7-.7-79.2-48.2-79.2-48.2 0-55.6 37.7-55.6 76.6V448h-92.8V149.1h89.1v40.8h1.3c12.4-23.6 42.6-48.2 87.6-48.2 93.7 0 110.9 61.7 110.9 141.9V448z"
          />
        </svg>
      ),
    },
  ],
  disclaimer = (
    <>
      <strong>Mention légale :</strong> Ce site est un projet indépendant, non affilié à une institution publique,
      ni au service de vélos en libre-service de la Ville de Paris. Les données utilisées proviendront de sources
      ouvertes (Open Data Paris, etc.). « Vélib’ » est une marque déposée appartenant à ses titulaires respectifs.
    </>
  ),
}: {
  siteName?: string;
  projectNote?: string;
  year?: number;
  social?: SocialItem[];
  disclaimer?: ReactNode;
}) {
  return (
    <footer className="site-footer">
      <div className="footer-grid">
        <div>
          © <span>{year}</span> {siteName} — {projectNote}
        </div>

        {social?.length ? (
          <div className="social">
            {social.map((s) => (
              <Link
                key={s.href}
                href={s.href}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={s.label}
                className="social-item"
              >
                {s.icon}
              </Link>
            ))}
            {/* Libellé sous les icônes (premier item si présent) */}
            <div className="social-label">{social[0]?.label}</div>
          </div>
        ) : null}
      </div>

      <p className="disclaimer">{disclaimer}</p>
    </footer>
  );
}