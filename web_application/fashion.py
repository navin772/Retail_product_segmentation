import streamlit as st
from PIL import Image
import requests
import random
from io import BytesIO

from pymilvus import connections, Collection

COLLECTION_NAME = 'clothing'  # Collection name
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

collection = Collection(name=COLLECTION_NAME)
collection.load()

def get_items_for_categories(cloth_type, num_items):

    expr = f'cloth_type == "{cloth_type}"'
    res = collection.query(
        expr = expr,
        output_fields=['id', 'filepath', 'cloth_type'],
        limit=num_items)
    sorted_res = sorted(res, key=lambda k: k['id'])
    ids = []
    image_urls = []
    for hits in sorted_res:
        ids.append(hits['id'])
        image_urls.append(hits['filepath'])

    return ids, image_urls

def show_similar_items(cloth_type, embedding, filepath, similar_items):

    expr = f'cloth_type == "{cloth_type}" && filepath != "{filepath}"'
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    result = collection.search(
        data = [embedding],
        anns_field='image_embedding',
        param = search_params,
        expr=expr,
        limit=similar_items,
        output_fields=['id', 'filepath'],
    )

    similar_ids = []
    similar_image_urls = []
    seen_urls = set()

    for hits in result:
        for hit in hits:
            url = hit.entity.get("filepath")
            id_ = hit.entity.get("id")
            # similar_ids.append(hit.entity.get("id"))
            # similar_image_urls.append(hit.entity.get("filepath"))
            if url not in seen_urls:  # Check if URL is unique
                similar_ids.append(id_)
                similar_image_urls.append(url)
                seen_urls.add(url)

    # similar_ids = similar_ids[3:]
    # similar_image_urls = similar_image_urls[3:]

    # Check whether 2 image urls are same or not, if same remove one of them from the list of urls and ids
    

    return similar_ids, similar_image_urls

import streamlit as st
import random

def display_cloth_cards(cloth_type, num_items):

    # Initialize session state for each card's Buy button
    if 'buy_clicked' not in st.session_state:
        st.session_state.buy_clicked = {}

    if 'similar_clicked' not in st.session_state:
        st.session_state.similar_clicked = {}

    ids, image_urls = get_items_for_categories(cloth_type, num_items)
    cards_per_row = 4

    # Loop through the cards data and create a card for each
    for i in range(0, len(ids), cards_per_row):
        card_columns = st.columns(cards_per_row)  # Divide the row into columns

        for j, column in enumerate(card_columns):
            card_index = i + j

            if card_index < len(ids):
                card_id = ids[card_index]
                image_url = image_urls[card_index]

                # Load and display image
                column.image(image_url, caption=f'{cloth_type} - {card_id}', use_column_width=True)

                # Add buttons
                buy_button = column.button('Buy', key=random.random())
                similar_button = column.button('Show Similar', key=f'similar_{card_id}')

                # if buy_button:
                #     st.session_state.buy_clicked[card_id] = True

                if similar_button:

                    st.session_state.similar_clicked[card_id] = True
                    # Get embedding for the image
                    expr = f'id == {card_id}'
                    res = collection.query(expr=expr, output_fields=['image_embedding', 'filepath'], limit=1)
                    embedding = res[0]['image_embedding']
                    filepath = res[0]['filepath']

                    ids, image_urls = show_similar_items(cloth_type, embedding, filepath, 5)

                    # Display similar items
                    with st.expander("Similar Items") as expander:

                        for id in ids:
                            column, _ = st.columns([1, 4])
                            column.image(image_urls[ids.index(id)], caption=f'{cloth_type} - {id}', use_column_width=True)

                    # for url in image_urls:
                    #     column.image(url, use_column_width=True)



def main():

    st.title("Retail Fashion Store")
    
    gender = st.radio("Select gender", ["Male", "Female"])

    if gender == "Male":
        cloth = st.selectbox("Select Category", ["tshirt", "jeans", "casual-shirt", "formal-shirt", "formal-trousers"])

    elif gender == "Female":
        cloth = st.selectbox("Select Category", ["tops", "women-jeans", "sarees", "kurta"])

    # id, image_urls = get_items_for_categories(cloth, 10)
    display_cloth_cards(cloth, 20)

