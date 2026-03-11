import shutil
import os
from flask import request, render_template, redirect, session
from db import get_user_conn
from security import verify_password
from user_files import save_user_data, get_user_data_path

def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        with get_user_conn() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email=?", (email,)
            ).fetchone()

        if user and verify_password(password, user["password_hash"]):
            session.permanent = True
            session["user_id"] = email
            
            # 既にデータがあれば削除しない。なければ空のデータを作成する。
            user_data_path = get_user_data_path(email)
            working_file = os.path.join(user_data_path, "working.json")
            if not os.path.exists(working_file):
                save_user_data(email, {"documents": []})

            return redirect("/dashboard")

        return render_template("login.html", error="ログイン失敗")

    return render_template("login.html")
