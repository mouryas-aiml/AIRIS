/**
 * AIRIS — Aerial Intelligence for Reconstruction and Avian Surveillance
 * Centralized Brand & Geospatial Configuration (SIH Problem Statement 26158)
 */

export const BRAND = {
  id: 'airis',
  name: 'AIRIS',
  fullName: 'AIRIS — Aerial Intelligence for Reconstruction and Avian Surveillance',
  shortTitle: 'AIRIS Geospatial Command Center',
  tagline: 'Aerial Intelligence for Reconstruction and Avian Surveillance',
  subtagline: 'UAV Photogrammetry & 3D Geospatial Intelligence Platform',
  problemStatement: 'SIH Problem Statement 26158',
  missionName: 'Single-Pass Drone Video to Accurate 3D Model Generation System',
  flightCode: 'AIRIS-UAV-ZURICH-01',
  gnssDatum: 'WGS 84 / UTM Zone 32N',
  spatialPrecision: '< 0.02m Relative Accuracy',
  cloudPoints: '518,786 points',
  reconstructionEngine: 'AI Multi-View Depth & Pose Graph Fusion',
  version: '2.4.0-RTK',
  systemStatus: 'OPERATIONAL (READY)',
  author: 'AIRIS Autonomous Photogrammetry Team',
  copyright: '© 2026 AIRIS Intelligence Systems · SIH 26158'
};

/**
 * Returns the official geometric SVG logo for AIRIS:
 * An abstract aerial/avian silhouette integrated with a 3D terrain/grid geometry.
 * Minimal, geometric, and instantly recognizable at any scale.
 */
export function getAirisLogoSvg({ size = 32, showText = false, textClass = '' } = {}) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="${size}" height="${size}" class="airis-brand-mark" aria-hidden="true">
    <defs>
      <linearGradient id="airis-grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#00e5ff"/>
        <stop offset="60%" stop-color="#00b4d8"/>
        <stop offset="100%" stop-color="#3b82f6"/>
      </linearGradient>
      <linearGradient id="airis-mesh" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#10b981" stop-opacity="0.9"/>
        <stop offset="100%" stop-color="#00e5ff" stop-opacity="0.3"/>
      </linearGradient>
    </defs>
    <!-- 3D Geodetic Isometric Base Grid -->
    <path d="M20 37 L5 28 L20 19 L35 28 Z" fill="#0b1728" stroke="#1e3a5f" stroke-width="1.2"/>
    <path d="M20 37 L20 19" stroke="#1e3a5f" stroke-width="0.8" stroke-dasharray="1.5 1.5"/>
    <path d="M12.5 32.5 L27.5 23.5" stroke="#1e3a5f" stroke-width="0.8" stroke-dasharray="1.5 1.5"/>
    <path d="M27.5 32.5 L12.5 23.5" stroke="#1e3a5f" stroke-width="0.8" stroke-dasharray="1.5 1.5"/>
    <!-- Abstract Aerial / Avian Delta-Wing Silhouette with 3D Facets -->
    <path d="M20 3 L35 25 L20 20 L5 25 Z" fill="url(#airis-grad)" filter="drop-shadow(0 2px 8px rgba(0,229,255,0.35))"/>
    <!-- Forward Central Keel / LiDAR Focal Vector -->
    <path d="M20 3 L20 20 L27.5 22.5 Z" fill="#ffffff" fill-opacity="0.22"/>
    <path d="M20 3 L20 20 L12.5 22.5 Z" fill="#000000" fill-opacity="0.18"/>
    <!-- Aerial Reconnaissance Apex Target Point -->
    <circle cx="20" cy="3" r="1.5" fill="#ffffff"/>
    <circle cx="20" cy="20" r="1.2" fill="#00e5ff"/>
  </svg>`;
  
  if (!showText) return svg;
  return `<span class="airis-brand-lockup">${svg}<span class="airis-brand-text ${textClass}"><strong>AIRIS</strong><small>GEOSPATIAL INTELLIGENCE</small></span></span>`;
}
