# Curated Notes from BaseCamp Sessions

---

## 🔧 GitHub & VSCode Setup

### **Common Issues and Fixes**

* **Cloning errors (e.g., BaseCamp2 not found)**: Participants reported that after forking the repo, `BaseCamp2` was missing.

  * **Fix**: Use the “sync fork” option on GitHub, then run `git pull` or `git pull --rebase` in VS Code terminal.
* **Using GitHub Desktop**: Some got access errors. The fix was to clone the correct repo:

  ```bash
  git clone https://github.com/outskill-git/GenAIEngineering-Cohort2.git
  ```

> 💬 *ChatGPT-style Tip*: “If your fork doesn't reflect upstream changes, go to GitHub and hit ‘Sync fork’. Then, do a `git pull` to update your local copy.”

---

## 🧠 Python Fundamentals & Environment Setup

### **Why Python for AI/ML?**

Common reasons shared:

* Rich libraries (NumPy, Pandas, TensorFlow)
* Easy syntax
* Interpreted, dynamically typed
* Huge community and support

> 💬 *ChatGPT-style Note*: “Python’s readability, extensive libraries, and low learning curve make it the go-to language for AI/ML.”

### **Virtual Environment (venv)**

* Create: `python -m venv <env_name>`
* Activate (Mac): `source <env_name>/bin/activate`
* Activate (VS-Code Git Bash): `source <env_name>/bin/activate`
* Activate (Windows CMD): `.\<env_name>\Scripts\activate.bat`
* Deactivate: `deactivate`
* Delete: Just remove the folder

> 💬 *ChatGPT-style Reminder*: “Always activate your virtual environment before installing packages. Use `.gitignore` to exclude `<env_name>/` from Git commits.”

---

## 💡 Programming Concepts Clarified

### **Python Nature**

* **Interpreted Language**: Executes line-by-line, no compilation needed.
* **No do-while Loop**: Simulate with:

  ```python
  while True:
      # logic
      if condition: break
  ```

### **Constants in Python**

* No true constants; use UPPER\_CASE variable names by convention.

> 💬 “Constants in Python rely on developer discipline—use all-caps and don’t reassign.”

### **Data Types and Casting**

* Dynamic typing allows flexibility, but beware of runtime type errors.
* Type hints (`-> int`) are non-enforced and meant for readability and tools like linters.

---

## 🔍 Python OOP Clarifications

### **Static Methods and Variables**

* Use decorators:

  ```python
  @staticmethod
  def my_static():
      pass
  ```

* Class variables are accessible via `ClassName.var_name`.

### **Function Overloading**

* Not supported. Use default arguments to simulate.

  ```python
  def greet(name="User"):
      print(f"Hello, {name}")
  ```

> 💬 “Python uses default values instead of traditional method overloading.”

---

## ⚙️ Tools & Extensions

### **Installing Packages**

* Use pip:

  ```bash
  pip install groq
  pip install ipykernel
  ```

* Update pip if facing installation issues:

  ```bash
  pip install --upgrade pip
  ```

### **Groq & Gemini APIs**

* API keys should be stored securely in a `.env` file.
* Use `.gitignore` to exclude `.env` from GitHub.

> 💬 “Never hardcode API keys—store them in a `.env` file and load using `dotenv`.”

---

## 🔐 .env and Security Best Practices

* **Storing Secrets**: Place keys in `.env` file and reference using `os.getenv()`.
* **Hiding from Git**: Add `.env` to `.gitignore`.

> 💬 “Using environment variables protects sensitive data and avoids hardcoding credentials.”

---

## 🧪 Development & Debugging Tips

* Use Python Notebooks (`.ipynb`) for development and `.py` files for production.
* To avoid memory issues, avoid loading very large files at once. Use `readline()` or `readlines()` with slicing.

---

## 📖 Additional Notes

* **Prompt Engineering** and **Gemini API Role Assignment** were briefly mentioned as follow-up areas.
* **Recordings & Slides** available via Circle community portal.

---
