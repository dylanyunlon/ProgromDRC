"""
Manual fix for crossprob compilation issues
"""

import os
import subprocess
import sys


def fix_source_files():
    """Fix missing headers in crossprob source files"""
    
    print("Fixing crossprob source files...")
    
    # Fix common.hh
    common_hh_path = "crossing-probability/src/common.hh"
    if os.path.exists(common_hh_path):
        with open(common_hh_path, 'r') as f:
            content = f.read()
        
        # Check if already fixed
        if '#include <iterator>' not in content:
            # Add missing includes after #include <string>
            content = content.replace(
                '#include <string>',
                '#include <string>\n#include <iterator>\n#include <limits>'
            )
            
            with open(common_hh_path, 'w') as f:
                f.write(content)
            print("✓ Fixed common.hh")
    
    # Fix common.cc
    common_cc_path = "crossing-probability/src/common.cc"
    if os.path.exists(common_cc_path):
        with open(common_cc_path, 'r') as f:
            content = f.read()
        
        # Add missing includes
        if '#include <limits>' not in content:
            content = content.replace(
                '#include <algorithm>',
                '#include <algorithm>\n#include <limits>'
            )
        
        # Fix numeric_limits usage
        content = content.replace(
            'numeric_limits<double>::infinity()',
            'std::numeric_limits<double>::infinity()'
        )
        
        with open(common_cc_path, 'w') as f:
            f.write(content)
        print("✓ Fixed common.cc")
    
    # Check for other potential issues
    # Fix any other files that might have similar issues
    src_dir = "crossing-probability/src"
    if os.path.exists(src_dir):
        for filename in os.listdir(src_dir):
            if filename.endswith(('.cc', '.hh', '.cpp', '.hpp')):
                filepath = os.path.join(src_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Fix common issues
                    original_content = content
                    
                    # Fix unqualified numeric_limits
                    if 'numeric_limits' in content and 'std::numeric_limits' not in content:
                        content = content.replace('numeric_limits', 'std::numeric_limits')
                    
                    # Fix unqualified ostream_iterator
                    if 'ostream_iterator' in content and 'std::ostream_iterator' not in content:
                        content = content.replace('ostream_iterator', 'std::ostream_iterator')
                    
                    if content != original_content:
                        with open(filepath, 'w') as f:
                            f.write(content)
                        print(f"✓ Fixed {filename}")
                        
                except Exception as e:
                    print(f"Warning: Could not process {filename}: {e}")


def main():
    """Main function to fix and install crossprob"""
    
    # Check if we're in the right directory
    if not os.path.exists("crossing-probability"):
        print("Error: crossing-probability directory not found!")
        print("Please run this script from the directory containing the cloned repository.")
        return False
    
    # Change to the repository directory
    os.chdir("crossing-probability")
    
    # Fix source files
    fix_source_files()
    
    # Try to build again
    print("\nAttempting to build crossprob...")
    
    # Clean first
    subprocess.run(["make", "clean"], capture_output=True)
    
    # Build
    result = subprocess.run(["make"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Build failed. Error output:")
        print(result.stderr)
        return False
    
    print("✓ Build successful")
    
    # Build Python extension
    print("\nBuilding Python extension...")
    result = subprocess.run(["make", "python"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Python extension build failed. Error output:")
        print(result.stderr)
        return False
    
    print("✓ Python extension built")
    
    # Install
    print("\nInstalling Python module...")
    result = subprocess.run([sys.executable, "setup.py", "install"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Installation failed, trying with --user flag...")
        result = subprocess.run([sys.executable, "setup.py", "install", "--user"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Installation failed. Error output:")
            print(result.stderr)
            return False
    
    print("✓ Installation successful")
    
    # Change back to parent directory
    os.chdir("..")
    
    # Test import
    print("\nTesting crossprob import...")
    try:
        import crossprob
        print("✓ crossprob imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import crossprob: {e}")
        return False


if __name__ == "__main__":
    # First check if crossprob is already installed
    try:
        import crossprob
        print("crossprob is already installed!")
        sys.exit(0)
    except ImportError:
        pass
    
    # Clone repository if not exists
    if not os.path.exists("crossing-probability"):
        print("Cloning crossing-probability repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/mosco/crossing-probability.git"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Failed to clone repository: {result.stderr}")
            sys.exit(1)
    
    # Run the fix
    success = main()
    
    if success:
        print("\n✓ crossprob has been successfully installed!")
        # Clean up
        import shutil
        shutil.rmtree("crossing-probability")
    else:
        print("\n✗ Failed to install crossprob")
        print("The crossing-probability directory has been kept for manual inspection")
        sys.exit(1)
