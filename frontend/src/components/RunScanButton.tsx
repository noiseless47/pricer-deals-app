interface RunScanButtonProps {
  isRunning: boolean
  onRun: () => void
}

export default function RunScanButton({ isRunning, onRun }: RunScanButtonProps) {
  return (
    <div className="run-scan">
      <button
        className="run-scan-button"
        onClick={onRun}
        disabled={isRunning}
        aria-busy={isRunning}
      >
        {isRunning ? (
          <>
            <span className="spinner" aria-hidden="true" />
            Scan in progress
          </>
        ) : (
          <>
            <span aria-hidden="true">↗</span>
            Start a deal scan
          </>
        )}
      </button>
    </div>
  )
}
