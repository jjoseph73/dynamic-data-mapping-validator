### 3. Setup Validation Script (scripts/validate_setup.sh)

```bash
#!/bin/bash

# Validate the project setup

set -e

ERRORS=0

check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Missing file: $1"
        ((ERRORS++))
    else
        echo "✅ Found: $1"
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "❌ Missing directory: $1"
        ((ERRORS++))
    else
        echo "✅ Found: $1"
    fi
}

check_executable() {
    if [ ! -x "$1" ]; then
        echo "❌ Not executable: $1"
        ((ERRORS++))
    else
        echo "✅ Executable: $1"
    fi
}

echo "🔍 Validating project setup..."

# Check essential files
echo ""
echo "📄 Checking essential files:"
check_file "docker-compose.yml"
check_file "Dockerfile"
check_file "requirements.txt"
check_file ".env"
check_file "README.md"

# Check directories
echo ""
echo "📁 Checking directories:"
check_dir "src"
check_dir "sql"
check_dir "mappings"
check_dir "scripts"

# Check scripts
echo ""
echo "🔧 Checking scripts:"
check_executable "deploy.sh"
check_executable "manage.sh"
check_executable "scripts/create_structure.sh"

# Check Docker files
echo ""
echo "🐳 Validating Docker configuration:"
if docker-compose config >/dev/null 2>&1; then
    echo "✅ docker-compose.yml is valid"
else
    echo "❌ docker-compose.yml has errors"
    ((ERRORS++))
fi

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
    echo "🎉 All checks passed! Setup is valid."
    exit 0
else
    echo "💥 Found $ERRORS error(s). Please fix them before proceeding."
    exit 1
fi
```
