from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model_decision_tree = pickle.load(open('model_decision_tree.pkl', 'rb'))
model_random_forest = pickle.load(open('model_random_forest.pkl', 'rb'))
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
enc = pickle.load(open('enc.pkl', 'rb'))


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sfh = int(request.form.get('sfh'))
    puw = int(request.form.get('puw'))
    ssl_final_state = int(request.form.get('ssl_final_state'))
    url_of_anchor = int(request.form.get('url_of_anchor'))
    urL_req = int(request.form.get('req_url'))
    url_length = int(request.form.get('url_length'))
    rank_traffic = int(request.form.get('rank'))
    is_ip_address = int(request.form.get('ip_address'))
    age_of_domain = int(request.form.get('age_of_domain'))

    if urL_req < 22:
        urL_req = 1
    elif urL_req < 61:
        urL_req = 0
    else:
        urL_req = -1

    if url_of_anchor < 31:
        url_of_anchor = 1
    elif url_of_anchor <= 67:
        url_of_anchor = 0
    else:
        url_of_anchor = -1

    if rank_traffic < 150000:
        rank_traffic = 1
    elif rank_traffic > 150000:
        rank_traffic = 0
    else:
        rank_traffic = -1

    if url_length < 54:
        url_length = 1
    elif url_length <= 75:
        url_length = 0
    else:
        url_length = -1

    if age_of_domain <= 6:
        age_of_domain = 1
    else:
        age_of_domain = -1

    input_values = [[sfh, puw, ssl_final_state, urL_req, url_of_anchor,
                     rank_traffic, url_length, age_of_domain, is_ip_address]]

    input_values = enc.transform(input_values)

    prediction_DS = model_decision_tree.predict(input_values)
    prediction_RF = model_random_forest.predict(input_values)
    prediction_SVM = model_svm.predict(input_values)

    result_map = {
        -1: "Phishy",
        0: "Suspicious",
        1: "Legit"
    }

    return render_template('index.html', prediction_DS=result_map[prediction_DS[0]], prediction_RF=result_map[prediction_RF[0]],prediction_SVM=result_map[prediction_SVM[0]] )


if __name__ == '__main__':
    app.run(debug=True)