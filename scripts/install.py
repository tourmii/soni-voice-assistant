import os
from pathlib import Path
import platform
from shutil import which
import subprocess


def is_uv_installed() -> bool:
    """
    Check if the UV tool is installed on the system.

    Returns:
        bool: True if the 'uv' command is available in the system path, False otherwise.
    """
    return which("uv") is not None


def install_uv() -> None:
    """
    Install the UV package management tool across different platforms.

    This function checks if UV is already installed. If not, it performs the installation:
    - On Windows, it uses pip to install UV and then updates it
    - On other platforms, it uses a curl-based installation script from Astral.sh

    Raises:
        subprocess.CalledProcessError: If the installation or update commands fail
    """
    if is_uv_installed():
        print("UV is already installed")
        return

    print("Installing UV...")
    if platform.system() == "Windows":
        subprocess.run(["pip", "install", "uv"])
        subprocess.run(["uv", "self", "update"])
    else:
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)
        subprocess.run(["uv", "self", "update"])


def main() -> None:
    """
    Set up the project development environment by installing UV, creating a virtual environment,
    and preparing the project for development.

    This function performs the following steps:
    1. Changes the current working directory to the project root
    2. Installs the UV package management tool
    3. Creates a Python 3.12.8 virtual environment
    4. Detects CUDA availability
    5. Installs the project in editable mode with appropriate dependencies
    6. Downloads and verifies project model files

    The function handles different platform-specific configurations and supports both CUDA and CPU-only installations.

    Notes:
        - Requires UV package manager to be available
        - Assumes project is structured with a standard Python project layout
        - Modifies system environment variables during execution
    """
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Install UV
    install_uv()

    # Create virtual environment
    subprocess.run(["uv", "venv", "--python", "3.12.8"])

    # Determine if CUDA is available
    if platform.system() == "Windows":
        venv_python = ".venv/bin/python"
    else:
        venv_python = ".venv/bin/python"
        os.environ["PATH"] = f"{os.path.dirname(venv_python)}:{os.environ['PATH']}"

    try:
        has_cuda = subprocess.run(["nvcc", "--version"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        has_cuda = False

    extras = "[cuda]" if has_cuda else "[cpu]"

    # Install project in editable mode
    env = os.environ.copy()
    env["PATH"] = f"{os.path.abspath('.venv/bin')}:{env['PATH']}"
    os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")
    os.system(f"uv pip install -e .{extras}")

    # Download and verify model files
    os.system("uv run glados download")


if __name__ == "__main__":
    main()
