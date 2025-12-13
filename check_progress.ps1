# Progress Checker for Historical Growth Analysis
# Run this script to check the progress of the analysis

$resultsFile = "data\historical_growth_results.json"

if (Test-Path $resultsFile) {
    try {
        $results = Get-Content $resultsFile -Raw | ConvertFrom-Json
        
        Write-Host "=" * 80
        Write-Host "Historical Growth Analysis - Progress Report"
        Write-Host "=" * 80
        Write-Host "Total Results Found: $($results.Count)"
        Write-Host "Results File: $resultsFile"
        Write-Host ""
        
        # Count unique tickers
        $uniqueTickers = $results | Select-Object -ExpandProperty ticker -Unique
        Write-Host "Unique Tickers Processed: $($uniqueTickers.Count) / 257"
        Write-Host ""
        
        # Group by event type
        Write-Host "Results by Event Type:"
        Write-Host "-" * 80
        $eventTypes = @("product_launch", "acquisition", "earnings_report", "partnership", "negative_event")
        
        foreach ($eventType in $eventTypes) {
            $eventResults = $results | Where-Object { $_.event_type -eq $eventType }
            if ($eventResults) {
                $count = $eventResults.Count
                $avg7d = ($eventResults | Where-Object { $_.avg_growth_7d -ne $null } | Measure-Object -Property avg_growth_7d -Average).Average
                $avg30d = ($eventResults | Where-Object { $_.avg_growth_30d -ne $null } | Measure-Object -Property avg_growth_30d -Average).Average
                
                Write-Host "$eventType`:"
                Write-Host "  Tickers: $count"
                if ($avg7d) { Write-Host "  Avg 7-Day Growth: $([math]::Round($avg7d, 2))%" }
                if ($avg30d) { Write-Host "  Avg 30-Day Growth: $([math]::Round($avg30d, 2))%" }
                Write-Host ""
            }
        }
        
        # Show last 10 processed tickers
        Write-Host "Last 10 Processed Tickers:"
        Write-Host "-" * 80
        $lastTickers = $results | Select-Object ticker, event_type -Unique | Group-Object ticker | Select-Object -Last 10
        foreach ($ticker in $lastTickers) {
            Write-Host "  $($ticker.Name)"
        }
        
        Write-Host ""
        Write-Host "=" * 80
        Write-Host "File Last Modified: $(Get-Item $resultsFile).LastWriteTime"
        Write-Host "=" * 80
        
    } catch {
        Write-Host "Error reading results file: $_"
        Write-Host "File may still be writing or corrupted."
    }
} else {
    Write-Host "Results file not found: $resultsFile"
    Write-Host "The analysis may still be initializing..."
    Write-Host "Check if Python process is running: Get-Process python"
}

