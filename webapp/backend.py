from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/",methods=['POST','GET'])

def index():
    if request.method == 'POST':
        csv_file = request.form['csvfile']
        csv_freq = request.form['csvfreq']
        runtimecost = request.form['runtimecost']
        replacementcost = request.form['replacecost']

    else:
        return render_template('frontpage.html')


if __name__ =="__main__":
    app.run(debug=True)