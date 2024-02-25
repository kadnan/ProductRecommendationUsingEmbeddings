from flask import Flask, render_template
from functions import *

app = Flask(__name__)


@app.route('/')
def index():
    products = get_all_products()
    return render_template('index.html', products=products)


@app.route('/product/<string:product_id>')
def product(product_id):
    product = get_product_by_id(product_id)
    if product:
        title = product['Title']
        top5 = fetch_similar_products(title)
        return render_template('product.html', product=product, top5=top5)
    else:
        return "Product not found", 404


if __name__ == '__main__':
    app.run(debug=True)
