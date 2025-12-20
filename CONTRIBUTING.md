# Contributing to Env-Doctor

Thank you for your interest in contributing to Env-Doctor! accurate environment detection is a hard problem, and we appreciate your help in making it better.

## 🛠️ Development Setup

To look at the code or fix a bug, clone the repo and set up a dev environment:

```bash
# 1. Clone the repository
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor

# 2. Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in editable mode
pip install -e .
```

Now you can run `env-doctor` directly from your terminal, and changes in the code will be reflected immediately.

## 🐛 Reporting Bugs

If you find a bug or a misdiagnosis (e.g., it says your driver is incompatible when it works fine), please open an issue with:
1.  Your OS & GPU Driver Version.
2.  The output of `env-doctor check`.
3.  The specific error you are facing.

## 🧪 Running Tests

We use `pytest`. Please ensure all tests pass before submitting a PR.

```bash
pip install pytest
pytest
```

## 📝 Pull Request Process

1.  Fork the repository and create a new branch (`git checkout -b feature/amazing-feature`).
2.  Commit your changes (`git commit -m 'Add some amazing feature'`).
3.  Push to the branch (`git push origin feature/amazing-feature`).
4.  Open a Pull Request.

Please make sure your code follows the existing style (clean, minimal dependencies).
