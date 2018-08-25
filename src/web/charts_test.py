from flask import Flask, Markup, render_template

app = Flask(__name__)

labels = ['ADVE', 'Email' ,'Form','Letter','Memo','News','Note','Report','Resume','Scientific']

values = [
    967.67, 1190.89, 1079.75, 1349.19,
    2328.91, 2504.28, 2873.83, 4764.87,
    4349.29, 6458.30, 9907, 16297
]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

@app.route('/bar')
def bar():
    bar_labels=labels
    bar_values=values
    return render_template('bar_chart.html', title='Predictions', max=7000, labels=bar_labels, values=bar_values)


if __name__ == '__main__':
    app.debug = True
    app.run(port=8080, debug=True)