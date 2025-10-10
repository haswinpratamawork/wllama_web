import { useMemo, useState } from 'react';
import {
  formatDuration as formatDurationMs,
  getPerfLabel,
  usePerfMonitor,
  type PerfEvent,
} from '../utils/perf-monitor';

const DEFAULT_KEYS = [
  'embedding:generate',
  'rag:search',
  'hnsw:search',
  'answer:generate',
];

type PerfMonitorPanelProps = {
  highlightKeys?: string[];
  maxRecent?: number;
  className?: string;
  hideHeader?: boolean;
  defaultShowDetails?: boolean;
  showToggle?: boolean;
};

export function PerfMonitorPanel({
  highlightKeys = DEFAULT_KEYS,
  maxRecent = 6,
  className = '',
  hideHeader = false,
  defaultShowDetails = false,
  showToggle = true,
}: PerfMonitorPanelProps) {
  const perfState = usePerfMonitor();
  const [showDetails, setShowDetails] = useState(defaultShowDetails);

  const summaryItems = useMemo(() => {
    return highlightKeys
      .map((key) => {
        const event = perfState.latestByName[key];
        if (!event) return null;
        return { key, event };
      })
      .filter(
        (item): item is { key: string; event: PerfEvent } => item !== null
      );
  }, [highlightKeys, perfState.latestByName]);

  const recentEvents = useMemo(() => {
    if (!showDetails) return [];
    return [...perfState.events].reverse().slice(0, maxRecent);
  }, [perfState.events, maxRecent, showDetails]);

  const hasContent = summaryItems.length > 0 || perfState.events.length > 0;

  const toggleButton = showToggle ? (
    <button
      type="button"
      className="btn btn-xs"
      onClick={() => setShowDetails((prev) => !prev)}
    >
      {showDetails ? 'Sembunyikan detail' : 'Lihat detail'}
    </button>
  ) : null;

  return (
    <div
      className={`bg-base-200 border border-base-300 rounded-xl p-4 space-y-3 ${className}`}
    >
      {!hideHeader ? (
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-base-content">
              Performance Monitor
            </h3>
            <p className="text-xs text-base-content/70">
              Pantau durasi embedding, RAG, dan jawaban model.
            </p>
          </div>
          {toggleButton}
        </div>
      ) : (
        showToggle && <div className="flex justify-end">{toggleButton}</div>
      )}

      {summaryItems.length > 0 ? (
        <div className="grid grid-cols-2 gap-3 text-sm">
          {summaryItems.map(({ key, event }) => (
            <div
              key={key}
              className="rounded-lg border border-base-300 bg-base-100 p-3 space-y-1"
            >
              <div className="text-xs uppercase tracking-wide text-base-content/60">
                {getPerfLabel(key)}
              </div>
              <div className="text-lg font-semibold text-base-content">
                {formatDurationMs(event.durationMs)}
              </div>
              <div className="text-[0.65rem] text-base-content/60">
                Update:{' '}
                {new Date(event.wallClock).toLocaleTimeString(undefined, {
                  hour12: false,
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                })}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-xs text-base-content/60">
          Belum ada data performa. Kirim prompt untuk mulai mengisi.
        </div>
      )}

      {showDetails && recentEvents.length > 0 && (
        <div className="text-xs space-y-2">
          <div className="font-semibold text-base-content">
            Riwayat terbaru
          </div>
          <div className="space-y-2 max-h-40 overflow-auto pr-1">
            {recentEvents.map((event) => (
              <div
                key={event.id}
                className="border border-base-300 rounded-lg p-2 bg-base-100"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-base-content">
                    {getPerfLabel(event.name)}
                  </span>
                  <span className="text-base-content/60">
                    {formatDurationMs(event.durationMs)}
                  </span>
                </div>
                <div className="text-[0.65rem] text-base-content/60">
                  {new Date(event.wallClock).toLocaleTimeString(undefined, {
                    hour12: false,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  })}
                </div>
                {event.meta && Object.keys(event.meta).length > 0 && (
                  <div className="mt-1 text-[0.65rem] text-base-content/70 whitespace-pre-wrap">
                    {Object.entries(event.meta)
                      .map(([metaKey, metaValue]) => {
                        if (metaValue === undefined || metaValue === null)
                          return `${metaKey}: -`;
                        if (typeof metaValue === 'object') {
                          try {
                            return `${metaKey}: ${JSON.stringify(metaValue)}`;
                          } catch {
                            return `${metaKey}: [object]`;
                          }
                        }
                        return `${metaKey}: ${metaValue}`;
                      })
                      .join('\n')}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {!hasContent && (
        <div className="text-xs text-base-content/60">
          Sistem monitoring siap digunakan.
        </div>
      )}
    </div>
  );
}

export default PerfMonitorPanel;
