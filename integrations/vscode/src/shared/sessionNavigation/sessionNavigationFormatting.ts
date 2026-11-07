/**
 * Format Sagent's native journal mtime for compact navigation surfaces.
 *
 * This is presentation-only; ordering and the timestamp itself remain owned by
 * Sagent's SessionInfo/list_sessions implementation.
 */
export function formatSessionAge(mtime: number, now = Date.now()): string {
  const seconds = Math.max(0, Math.floor((now - mtime * 1000) / 1000));
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  return `${Math.floor(hours / 24)}d ago`;
}
