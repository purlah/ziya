### To Run locally
First time run of build script should install all dependencies
```bash
python ./ziya_build.py
```
To just test FrontEnd changes run
```bash
poetry run fbuild
PYTHONPATH=$(pwd) ZIYA_LOG_LEVEL=DEBUG poetry run python app/main.py --port 6868

Run with aws profile: 
poetry run fbuild && poetry run python app/main.py --profile ziya --port 6868
```

#### To test Backend and FrontEnd changes via locally installed pip file run
```bash
pip uninstall ziya -y
python ./ziya_build.py
pip install dist/
```

#### To run unit tests for backend
```bash
poetry run pytest
```

#### To run diff application pipeline tests
```bash
python tests/run_diff_tests.py --multi
```

#### To publish to PyPi:
```bash
python ./ziya_build.py && poetry publish
pip install ziya --upgrade
OR 
pipx upgrade ziya
```

### FAQ
#### To install a specific version of a package
```bash
pip install ziya==0.1.3
```

#### To publish and test in the testpypi repository:
```bash
python ./ziya_build.py
poetry publish --repository testpypi
pip uninstall ziya -y
pip install --index-url https://test.pypi.org/simple/ ziya
```

### Breakdown of manual build steps:
cd ziya-clone
poetry lock
poetry install --with dev
cd frontend
npm install
cd ..
mkdir templates
poetry run fbuild
# note that the wheel is manually populated with templates and forced to platform independence

### Special considerations:
# if you are using ziya to edit itself, you may need to override certain sentinels
ZIYA_TOOL_SENTINEL=anythingUnlikely  
