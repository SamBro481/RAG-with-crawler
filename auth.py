# auth.py
import os
import hashlib
import uuid
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def sign_up(username: str, password: str) -> dict:
    # Check if username already exists
    existing = supabase.table("users")\
        .select("id")\
        .eq("username", username)\
        .execute()

    if existing.data:
        return {"success": False, "error": "Username already exists"}

    # Create new user
    result = supabase.table("users").insert({
        "username": username,
        "password_hash": hash_password(password)
    }).execute()

    user = result.data[0]
    return {"success": True, "user_id": user["id"], "username": user["username"]}


def login(username: str, password: str) -> dict:
    result = supabase.table("users")\
        .select("*")\
        .eq("username", username)\
        .eq("password_hash", hash_password(password))\
        .execute()

    if not result.data:
        return {"success": False, "error": "Invalid username or password"}

    user = result.data[0]
    return {"success": True, "user_id": user["id"], "username": user["username"]}


def save_crawled_site(user_id: str, url: str, pages_count: int, chunks_count: int):
    # Check if this URL was already crawled by this user
    existing = supabase.table("crawled_sites")\
        .select("id")\
        .eq("user_id", user_id)\
        .eq("url", url)\
        .execute()

    if existing.data:
        # Update existing record
        supabase.table("crawled_sites")\
            .update({
                "pages_count": pages_count,
                "chunks_count": chunks_count,
                "crawled_at": "NOW()"
            })\
            .eq("user_id", user_id)\
            .eq("url", url)\
            .execute()
    else:
        # Insert new record
        supabase.table("crawled_sites").insert({
            "user_id": user_id,
            "url": url,
            "pages_count": pages_count,
            "chunks_count": chunks_count
        }).execute()
        
def delete_crawled_site(user_id: str, url: str):
    supabase.table("crawled_sites")\
        .delete()\
        .eq("user_id", user_id)\
        .eq("url", url)\
        .execute()


def get_crawled_sites(user_id: str) -> list:
    result = supabase.table("crawled_sites")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("crawled_at", desc=True)\
        .execute()

    return result.data
