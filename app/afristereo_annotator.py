import streamlit as st
import pandas as pd
import uuid
import json
import copy
import logging
from sklearn.decomposition import PCA
import numpy as np
import sys

# === Command-line Argument Parsing ===
default_src = "./data/processed/attribute_to_group_initial.json"
default_dst = "./data/processed/attribute_to_group_modified.json"

src_path = default_src
dst_path = default_dst

for arg in sys.argv:
    if arg.startswith("--src="):
        src_path = arg.split("=", 1)[1]
    elif arg.startswith("--dst="):
        dst_path = arg.split("=", 1)[1]

# === Semantic Sorting Utility ===
def sort_df_by_semantic_similarity(df, embedding_path="./app/app_data/attribute_embeddings.json", group_col='group'):
    with open(embedding_path, "r") as f:
        embedding_dict = json.load(f)

    group_embeddings = []
    for group in df[group_col]:
        group_embeds = [np.array(embedding_dict[term]) for term in group if term in embedding_dict]
        if group_embeds:
            centroid = np.mean(group_embeds, axis=0)
        else:
            centroid = np.zeros(len(next(iter(embedding_dict.values()))))
        group_embeddings.append(centroid)

    pca = PCA(n_components=1)
    group_1d = pca.fit_transform(group_embeddings).flatten()
    df['sort_key'] = group_1d
    df_sorted = df.sort_values(by='sort_key').drop(columns='sort_key').reset_index(drop=True)
    return df_sorted

# === Streamlit Config ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")
st.title("Attribute Group Annotator (Group-Only View)")

# === Load Initial Data ===
@st.cache_data
def load_groups_from_attribute_json():
    with open(src_path, "r") as f:
        raw = json.load(f)
    seen = set()
    rows = []
    for group in raw.values():
        group_tuple = tuple(sorted(group))
        if group_tuple not in seen:
            seen.add(group_tuple)
            rows.append({
                "group": list(group_tuple),
                "group_str": ", ".join(group_tuple),
                "row_id": str(uuid.uuid4()),
                "selected": False,
                "flag": "⚪️"
            })
    df = pd.DataFrame(rows)
    return sort_df_by_semantic_similarity(df)

# === Session State ===
if "df" not in st.session_state:
    st.session_state.df = load_groups_from_attribute_json()
    logger.info("✅ Group data loaded and sorted.")

if "df_history" not in st.session_state:
    st.session_state.df_history = []

full_df = st.session_state.df

# === Flag Filtering ===
st.subheader("🔍 Filter by Flag")
flag_filter = st.radio("Choose flag to filter:", ["All", "⚪️", "🟥", "🟨", "🟩", "🔵"], horizontal=True)
if flag_filter == "All":
    df = full_df.copy()
else:
    df = full_df[full_df["flag"] == flag_filter].copy()

# === Display Table ===
edited_df = st.data_editor(
    df[["selected", "group_str", "flag"]],
    key="group_editor",
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True
)

# === Flag Management ===
st.subheader("🏳️ Add Flag to Selected Groups")
flag_choice = st.selectbox("Choose a flag to assign:", ["🟥", "🟨", "🟩", "🔵"])
if st.button("Apply Flag"):
    selected_ids = df["row_id"][edited_df["selected"]]
    full_df.loc[full_df["row_id"].isin(selected_ids), "flag"] = flag_choice
    st.session_state.df = sort_df_by_semantic_similarity(full_df)
    st.rerun()

# === Merge Groups ===
st.subheader("🔗 Merge Selected Groups")
if st.button("Merge Selected Groups"):
    selected = df[df["row_id"].isin(df["row_id"][edited_df["selected"]])]

    if len(selected) >= 2:
        st.session_state.df_history.append(copy.deepcopy(full_df))

        merged_group = set()
        for g in selected["group"]:
            merged_group.update(g)

        flag_counts = selected["flag"].value_counts()
        majority_flag = flag_counts.idxmax() if not flag_counts.empty else "⚪️"

        new_row = {
            "group": list(merged_group),
            "group_str": ", ".join(sorted(merged_group)),
            "row_id": str(uuid.uuid4()),
            "selected": False,
            "flag": majority_flag
        }

        full_df = full_df[~full_df["row_id"].isin(selected["row_id"])]
        full_df = pd.concat([pd.DataFrame([new_row]), full_df], ignore_index=True)
        st.session_state.df = sort_df_by_semantic_similarity(full_df)

        st.success(f"✅ Merged selected groups (assigned flag: '{majority_flag}').")
        st.rerun()

# === Separate Attributes ===
st.subheader("✂️ Separate Attributes from Groups")

if "rows_to_edit" not in st.session_state:
    st.session_state.rows_to_edit = []

if st.button("Separate Attributes"):
    selected_rows = df[df["row_id"].isin(df["row_id"][edited_df["selected"]])]

    if selected_rows.empty:
        st.warning("⚠️ No groups selected.")
    else:
        st.session_state.df_history.append(copy.deepcopy(full_df))
        st.session_state.rows_to_edit = selected_rows.head(10).to_dict(orient="records")
        st.session_state.attrs_to_remove = {
            row["row_id"]: set() for row in st.session_state.rows_to_edit
        }

if st.session_state.get("rows_to_edit"):
    st.write("### Select items to remove from group (split into separate rows):")
    rows_to_edit = st.session_state.rows_to_edit
    for row in rows_to_edit:
        st.markdown(f"**Group:**")
        items = row["group"]
        cols = st.columns(len(items) + 1)
        for i, item in enumerate(items):
            key = f"remove_{row['row_id']}_{item}"
            checked = cols[i].checkbox(f"❌ {item}", key=key)
            if checked:
                st.session_state.attrs_to_remove[row["row_id"]].add(item)
            else:
                st.session_state.attrs_to_remove[row["row_id"]].discard(item)

    if st.button("Confirm Removal"):
        updated_rows = []
        new_rows = []

        for row in rows_to_edit:
            to_remove = st.session_state.attrs_to_remove.get(row["row_id"], set())
            group = row["group"]
            keep = [g for g in group if g not in to_remove]
            original_flag = row.get("flag", "⚪️")

            if keep:
                updated_rows.append({
                    "group": keep,
                    "group_str": ", ".join(keep),
                    "row_id": row["row_id"],
                    "selected": False,
                    "flag": original_flag
                })

            for rem in to_remove:
                new_rows.append({
                    "group": [rem],
                    "group_str": rem,
                    "row_id": str(uuid.uuid4()),
                    "selected": False,
                    "flag": original_flag
                })

        full_df = full_df[~full_df["row_id"].isin([r["row_id"] for r in rows_to_edit])]
        full_df = pd.concat([pd.DataFrame(updated_rows + new_rows), full_df], ignore_index=True)
        st.session_state.df = sort_df_by_semantic_similarity(full_df)

        del st.session_state.rows_to_edit
        del st.session_state.attrs_to_remove
        st.success("✅ Removed items and split into new rows (flag preserved).")
        st.rerun()

# === Undo ===
if st.button("↩️ Undo Last Change"):
    if st.session_state.df_history:
        df_previous = st.session_state.df_history.pop()
        st.session_state.df = sort_df_by_semantic_similarity(df_previous)
        st.success("↩️ Reverted to previous state.")
        st.rerun()
    else:
        st.warning("⚠️ No previous change to undo.")

# === Save ===
if st.button(f"💾 Save to {dst_path}"):
    save_dict = {}
    for _, row in st.session_state.df.iterrows():
        for attr in row["group"]:
            save_dict[attr] = row["group"]
    with open(dst_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    st.success(f"✅ Saved to {dst_path}")