#!/usr/bin/env node

const { spawn } = require('child_process');

function parseArgs(argv) {
  const args = {
    modelPath: 'artifacts/models/svc_cut_model.joblib',
    pythonCmd: process.env.PYTHON_CMD || 'python',
    json: null,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!value) continue;

    if (key === '--model-path') {
      args.modelPath = value;
      i += 1;
    } else if (key === '--python-cmd') {
      args.pythonCmd = value;
      i += 1;
    } else if (key === '--json') {
      args.json = value;
      i += 1;
    }
  }

  return args;
}

function runPrediction({ pythonCmd, modelPath, json }) {
  return new Promise((resolve, reject) => {
    if (!json) {
      reject(
        new Error(
          'Missing --json input. Example: --json "{\"carat\":0.7,\"color\":\"E\",...}"'
        )
      );
      return;
    }

    const cmdArgs = [
      'predict_svc.py',
      '--model-path',
      modelPath,
      '--json',
      json,
    ];

    const child = spawn(pythonCmd, cmdArgs, {
      cwd: process.cwd(),
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', (err) => {
      reject(err);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Python process exited with code ${code}`));
        return;
      }

      const line = stdout
        .split('\n')
        .map((s) => s.trim())
        .find((s) => s.startsWith('Predicted cut:'));

      if (!line) {
        reject(new Error(`Unexpected output from predict_svc.py:\n${stdout}`));
        return;
      }

      const prediction = line.split(':').slice(1).join(':').trim();
      resolve({ prediction, rawOutput: stdout.trim() });
    });
  });
}

async function main() {
  try {
    const options = parseArgs(process.argv);
    const result = await runPrediction(options);
    console.log(`Prediction: ${result.prediction}`);
  } catch (err) {
    console.error(`Prediction failed: ${err.message}`);
    process.exit(1);
  }
}

main();
