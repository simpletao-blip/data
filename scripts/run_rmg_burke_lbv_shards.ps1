$ErrorActionPreference = "Stop"

$python = "C:\Users\30962\anaconda3\envs\cantera\python.exe"
$mechanism = "mechanisms\converted\RMG_2026_Burke_normalized_v2.yaml"
$env:PYTHONPATH = "C:\Users\30962\Documents\cantera\src"

$jobs = @()
for ($index = 0; $index -lt 4; $index++) {
    $arguments = @(
        "scripts\run_lbv_validation_with_timeout.py",
        "--design", "data\processed\lbv_rmg_burke_shard_$index.csv",
        "--mechanism", $mechanism,
        "--mechanism-id", "RMG_2026_Burke",
        "--output", "results\raw\RMG_2026_Burke_lbv_shard_$index.csv",
        "--case-timeout-s", "300",
        "--resume"
    )
    $jobs += Start-Job -Name "RMG_Burke_LBV_$index" -ScriptBlock {
        param($workingDirectory, $pythonExecutable, $pythonPath, $workerArguments)
        Set-Location $workingDirectory
        $env:PYTHONPATH = $pythonPath
        & $pythonExecutable @workerArguments
        if ($LASTEXITCODE -ne 0) {
            throw "LBV worker exited with code $LASTEXITCODE"
        }
    } -ArgumentList (Get-Location).Path, $python, $env:PYTHONPATH, (, $arguments)
}

$jobs | Wait-Job | Out-Null
foreach ($job in $jobs) {
    Receive-Job -Job $job | Out-File -FilePath "results\logs\$($job.Name).log" -Encoding utf8
}
$failed = $jobs | Where-Object { $_.State -ne "Completed" }
if ($failed) {
    $states = ($failed | ForEach-Object { "$($_.Name):$($_.State)" }) -join ", "
    throw "One or more LBV workers failed: $states"
}

& $python scripts\merge_lbv_shards.py `
    --design data\processed\lbv_validation_design.csv `
    --output results\raw\RMG_2026_Burke_lbv_validation_staged.csv `
    results\raw\RMG_2026_Burke_lbv_shard_0.csv `
    results\raw\RMG_2026_Burke_lbv_shard_1.csv `
    results\raw\RMG_2026_Burke_lbv_shard_2.csv `
    results\raw\RMG_2026_Burke_lbv_shard_3.csv
