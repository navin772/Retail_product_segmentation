import streamlit as st
import random

from pymilvus import connections, Collection

COLLECTION_NAME = 'clothing'  # Collection name
MILVUS_HOST = "milvus.default.svc.cluster.local"  # uses the milvus standalone deployed on the kubernetes cluster
MILVUS_PORT = "19530"

# Use this for local testing (milvus installed on local machine)
# COLLECTION_NAME = 'clothing'  # Collection name
# MILVUS_HOST = "localhost"  # uses the milvus standalone deployed on the kubernetes cluster
# MILVUS_PORT = "19530"

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

            if url not in seen_urls:  # Check if URL is unique
                similar_ids.append(id_)
                similar_image_urls.append(url)
                seen_urls.add(url)

    return similar_ids, similar_image_urls


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

                if similar_button:

                    st.session_state.similar_clicked[card_id] = True
                    # Get embedding for the image
                    expr = f'id == {card_id}'
                    res = collection.query(expr=expr, output_fields=['image_embedding', 'filepath'], limit=1)
                    embedding = res[0]['image_embedding']
                    filepath = res[0]['filepath']

                    ids, image_urls = show_similar_items(cloth_type, embedding, filepath, 5)

                    # Display similar items
                    with st.expander(":green[Show similar items]", expanded=True) as expander:
                        st.markdown("## Similar Items")

                        cols = st.columns(len(ids))
                        
                        for col, id in zip(cols, ids):
                            col.image(image_urls[ids.index(id)], caption=f'{cloth_type} - {id}', use_column_width=True)


# Determines the milvus DB mapping for the UI categories
def return_cloth_category(gender, cloth_in_ui) -> str:
    cloth_dict_men = {
        "Tshirts": "tshirt",
        "Jeans": "jeans",
        "Casual Shirts": "casual-shirt",
        "Formal Shirts": "formal-shirt",
        "Formal Trousers": "formal-trousers",
    }

    cloth_dict_women = {
        "Tops": "tops",
        "Jeans": "women-jeans",
        "Sarees": "sarees",
        "Kurtis": "kurta",
    }

    if gender == "Male":
        return cloth_dict_men[cloth_in_ui]
    
    elif gender == "Female":
        return cloth_dict_women[cloth_in_ui]


def main():

    st.image(image='https://assets.fxcm.com/cdn-cgi/image/quality=100,format=webp,fit=contain,width=828/fxpress/fxcmcom/uk/insight/AACoverImages/iStock-LuxuryFashion1.jpg', width=900)
    st.title("Retail Fashion Store")
    
    gender = st.radio("Select gender", ["Male", "Female"])

    if gender == "Male":
        cloth = st.selectbox("Select Category", ["Tshirts", "Jeans", "Casual Shirts", "Formal Shirts", "Formal Trousers"])

    elif gender == "Female":
        cloth = st.selectbox("Select Category", ["Tops", "Jeans", "Sarees", "Kurtis"])

    num_items = st.number_input("Number of items to display", min_value=1, max_value=100, value=20)

    st.write("Select the items you like and click on 'Show Similar' to see similar items")

    cloth_type = return_cloth_category(gender, cloth)

    display_cloth_cards(cloth_type, num_items)

