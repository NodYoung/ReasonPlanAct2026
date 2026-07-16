# Reason Plan Act: ROS 2 + LangChain Tutorial
This workspace includes a small ROS 2 chat package built around LangChain. The
environment is managed with [pixi](https://pixi.sh), which installs both ROS 2
(via [RoboStack](https://robostack.github.io/)) and the Python/LangChain stack
into a single, reproducible environment.

## 1. Install pixi

If you have a Linux/Mac machine
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
If you have a Windows machine
```bash
irm -useb https://pixi.sh/install.ps1 | iex
```

Then restart your shell (or run `source ~/.zshrc` on Mac or `source ~/.bashrc` on Linux) so the `pixi` command is on your `PATH`.

Verify pixi is available with:
```bash
pixi --version
```

## 2. Clone the git repository and open inside a terminal or your favourite IDE (VSCode etc)
```bash
git clone https://github.com/ReasonPlanAct/ReasonPlanAct2026.git
cd ReasonPlanAct2026
code . # (if using VSCode or similar to open the tutorial folder in your IDE)
```

## 3. Fill in the model provider details
Get your API key from Gemini, OpenAI, Ollama, or other provider and fill in the
`.env` file. You can copy the example file and edit it with VS Code:
```bash
# Move to the session 1 directory
cd session_1
# Copy example.env and rename to .env
cp example.env .env
# Open the .env file with vscode
code .env
```
### We will be using Google Gemini

You can follow the following instruction (accompanies video)
1. Navigate to aistudio.google.com (you need a google account - some Intitutional accounts may not allow this, use a personal account in that case)
2. At the bottom left, click `Get API Key' (key icon)
3. Top Right, click 'Create API Key'
4. Provide a name for your key and `Choose an imported project'. Don't worry about Cloud project details — AI Studio creates one for you automatically if you don't have one.
5. Click `Create Key'
6. A key will be generated and copy this to the clipboard

Note: Store the key safely: Treat it like a password — never commit it to git or hardcode it in frontend JavaScript.


https://github.com/user-attachments/assets/8d69fbf1-1c4c-4567-bf2f-fcab47c4db49


### Populate .env with your key

Once you have your Gemini key populate the .env file by pasting your copied key over 'your_key_here' next to GOOGLE_API_KEY.

## 4. Install the environment
From the repository root:
```bash
pixi install        # Install required dependencies using pixi
```

## 5. Run test commands
Use the following commands to run the tutorial and ROS nodes inside the pixi environment.

### ROS Turtlesim
```bash
pixi run turtlesim  # Launch the turtlesim window    
```
In some cases, you may get a notification asking to allow the app to find devices on local network, please allow it (moslty in Mac)

Control the turtle manually with the teleop node (in a second terminal):
```bash

pixi run teleop     # Run the teleop control for turtlesim
```

## 6. Running the tutorial. 

Launch the tutorial notebooks. This will open a Jupyter Lab session in your browser, where you can run the tutorial notebooks. This is the recommended way for running the tutorial. 

```bash
pixi run tutorial   # Run the tutorial notebooks in Jupyter Lab
```

NOTE: If you need to use VS Code instead of Jupyter Lab:
Open the notebooks in VS Code and select the "Python (tutorial_rpa)" kernel. 

### To reset
```bash
# Remove the pixi environment and reinstall
rm -rf .pixi
pixi install
```

