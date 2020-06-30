class NeuralNet {
	constructor(){
		this.numOfLayers = arguments.length
		this.numOfHiddensLayers = this.numOfLayers - 2
		this.numOfWeights = this.numOfLayers - 1
		this.numOfErrors = this.numOfLayers - 1
		
		this.layer = []
		this.layer.length = arguments.length
		this.weights = []
		this.weights.length = arguments.length - 1
		this.bias = []
		this.bias.length = arguments.length - 1
		this.errors = []
		this.errors.length = arguments.length - 1
		
		for(let i = 0; i < arguments.length - 1; i++){
			this.layer[i] = new Matrix(arguments[i], 1)
			this.weights[i] = new Matrix(arguments[i+1], arguments[i])
			this.bias[i] = new Matrix(arguments[i+1], 1)
			this.errors[i] = new Matrix(arguments[i+1], 1)
			
			this.weights[i].setRandom()
			this.bias[i].setRandom()
			
		}
		
		this.layer[arguments.length - 1] = new Matrix(arguments[arguments.length - 1], 1)
		this.layer[arguments.length - 1].setRandom()
		
		this.targets = new Matrix(arguments[arguments.length - 1], 1)
			
		this.lr = 0.5
	}
	
	
	test(a,b){
		this.layer[0].data[0][0] = a 
		this.layer[0].data[1][0] = b
		
		this.feedForward()
		console.log(`${a} ; ${b} ->  ${Math.round(this.layer[this.numOfLayers - 1].data[0][0] * 100) / 100}`)
	}
	
	training(iteration){	
		for(let i = 0; i < iteration; i++){
			this.layer[0].data[0][0] = Math.floor(Math.random()*2)
			this.layer[0].data[1][0] = Math.floor(Math.random()*2)
			if(1 * this.layer[0].data[0] + 1 * this.layer[0].data[1] === 1){
				this.targets.data[0][0] = 1
				//this.targets.data[1][0] = 0
				}else{
				this.targets.data[0][0] = 0
				//this.targets.data[1][0] = 1
				}
			this.feedForward()
			this.backpropagation()
		}
	}
	
	activtion = x => Math.exp(x)/(1 + Math.exp(x))
	
	derivativeActivtion = x => x * (1 - x)
	
	feedForward(){	
		for(let i = 1; i < this.numOfLayers; i++){
			this.layer[i] = this.layer[i].add(this.weights[i-1].multiply(this.layer[i-1]),this.bias[i])
			this.layer[i].data = this.layer[i].data.map(v => v.map(v => this.activtion(v)))
		}	
	}
	
	getErrors(){
		this.errors[this.numOfErrors - 1] = this.layer[this.numOfLayers - 1].subtraction(this.targets)
		
		for(let i = 0; i < this.numOfErrors - 1; i++){
			let wt = this.weights[this.numOfWeights - 1 - i].transpose()
			let err = this.errors[this.numOfErrors - 1 - i]
			this.errors[this.numOfErrors - 2 - i] = wt.multiply(err)
		}
	}
	
	backpropagation(){
		this.getErrors()
		
		for(let i = 0; i < this.numOfWeights; i++){
			let delta = new Matrix(this.layer[this.numOfLayers - 1 - i].rows, this.layer[this.numOfLayers - 1 - i].cols)
			delta.data = this.layer[this.numOfLayers - 1 - i].data.map((e,k) => e.map((v,j) => this.derivativeActivtion(v) * this.errors[this.numOfErrors - 1 - i].data[k][j] * 2 * this.lr)) //dSig * dCost * lr
			this.bias[this.numOfWeights - 1 - i] = this.bias[this.numOfWeights - 1 - i].subtraction(delta)
			delta = delta.multiply(this.layer[this.numOfLayers - 2 - i].transpose()) 
			this.weights[this.numOfWeights - 1 - i] = this.weights[this.numOfWeights - 1 - i].subtraction(delta)
		}
	}
}

class Matrix {
	constructor(rows, cols){
		this.rows = rows
		this.cols = cols
		this.data = new Array(rows)
		this.setRandom()
	}
	
	setRandom(){
		for(let i = 0; i < this.rows; i++){
			let arr = new Array(this.cols)
			this.data[i] = arr.fill(0)
			}
		this.data.forEach(item => {
		item.forEach((item, index, arr) =>  arr[index] = (Math.random() * 2 - 1 ))
		})		
	}
	
	subtraction(m2){
		if(this.cols === m2.cols && this.rows === m2.rows){
		let m3 = new Matrix(this.rows, this.cols)
		m3.data = this.data.map((e,i) => e.map((v,j) => v - m2.data[i][j]))
		return m3
		}else{throw "matrices is not the same size"}
	}
	
	add(m2){
		let m3 = new Matrix(this.rows, this.cols)
		if(typeof(m2) === "object"){
			m3.data = this.data.map((e,i) => e.map((v,j) => v + m2.data[i][j]))
		}else{
			m3.data = this.data.map((e,i) => e.map((v,j) => v + m2))
		}
		return m3
	}
	
	transpose(){
		let mt = new Matrix(this.cols, this.rows)
		
		for(let i in this.data){
			for(let j in this.data[i]){
				mt.data[j][i] = this.data[i][j]
			}
		}
		return mt
	}
	
	multiply(m2){
		let m3
		if (typeof(m2) === "object"){
		if (this.cols === m2.rows){
			m3 = new Matrix(this.rows, m2.cols)
			for(let i = 0; i < this.rows; i++){
				for(let j = 0; j < m2.cols; j++){
					let sum = 0
					for(let k = 0; k < this.cols; k++){
						sum += this.data[i][k] * m2.data[k][j]
					}
					m3.data[i][j] = sum

				}
			}
		}else{throw "cols not equal to rows"}	
		}else{
			m3 = new Matrix(this.rows, this.cols)
			for(let x in this.data){
				for(let y in this.data[x]){
					m3.data[x][y] = this.data[x][y] * m2
				}
			}
		}
		return m3
	}
	
	p = () => console.table(this.data)

}

let nn = new NeuralNet(2,6,6,1)
nn.training(10000)
t()
function t(){
nn.test(0,0)
nn.test(0,1)
nn.test(1,0)
nn.test(1,1)
}
