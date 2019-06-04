import React from 'react';
import { API_ROOT } from '../../api-config';
import { withRouter } from 'react-router-dom';
import HeatMap from '../HeatMap'
import TextSaliencyMap from '../Interpretation'
import Model from '../Model'
import OutputField from '../OutputField'
import {
  Accordion,
  AccordionItem,
  AccordionItemTitle,
  AccordionItemBody,
  } from 'react-accessible-accordion';
import '../../css/TeComponent.css';

const apiUrl = () => `${API_ROOT}/predict/textual-entailment`
const apiUrlInterpret = ({interpreter}) => `${API_ROOT}/interpret/textual-entailment/${interpreter}`

const title = "Textual Entailment"

const description = (
  <span>
    <span>
    Textual Entailment (TE) takes a pair of sentences and predicts whether the facts in the first
    necessarily imply the facts in the second one.  The AllenNLP toolkit provides the following TE visualization,
    which can be run for any TE model you develop.
    This page demonstrates a reimplementation of
    </span>
    <a href = "https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27" target="_blanke" rel="noopener noreferrer">{' '} the decomposable attention model (Parikh et al, 2017) {' '}</a>
    <span>
    , which was state of the art for
    </span>
    <a href = "https://nlp.stanford.edu/projects/snli/" target="_blank" rel="noopener noreferrer">{' '} the SNLI benchmark {' '}</a>
    <span>
    (short sentences about visual scenes) in 2016.
    Rather than pre-trained Glove vectors, this model uses <a href="https://arxiv.org/abs/1802.05365">ELMo embeddings</a>, which are completely character based and improve performance by 2%
    </span>
  </span>
  );

const descriptionEllipsed = (
  <span>
    Textual Entailment (TE) takes a pair of sentences and predicts whether the facts in the first necessarily imply the…
  </span>
)

const fields = [
  {name: "premise", label: "Premise", type: "TEXT_INPUT",
   placeholder: 'E.g. "A large, gray elephant walked beside a herd of zebras."'},
  {name: "hypothesis", label: "Hypothesis", type: "TEXT_INPUT",
   placeholder: 'E.g. "The elephant was lost."'}
]

const TeGraph = ({x, y}) => {
  const width = 224;
  const height = 194;

  const absoluteX = Math.round(x * width);
  const absoluteY = Math.round((1.0 - y) * height);

  const plotCoords = {
    left: `${absoluteX}px`,
    top: `${absoluteY}px`,
  };

  return (
    <div className="te-graph-labels">
    <div className="te-graph">
      <div className="te-graph__point" style={plotCoords}></div>
    </div>
    </div>
  )
}

const judgments = {
  CONTRADICTION: <span>the premise <strong>contradicts</strong> the hypothesis</span>,
  ENTAILMENT: <span>the premise <strong>entails</strong> the hypothesis</span>,
  NEUTRAL: <span>there is <strong>no correlation</strong> between the premise and hypothesis</span>
}

const getTokenWeightPairs = (premiseGrads, hypothesisGrads, premise_tokens, hypothesis_tokens, premiseTopK, hypothesisTopK) => {
  // map to objects with indices
  let premiseGradsWithIdx = premiseGrads.map((grad, idx) => { return {grad, idx} })
  console.log('PREMISE GRADS', premiseGradsWithIdx)
  let hypothesisGradsWithIdx = hypothesisGrads.map((grad, idx) => { return {grad, idx} })

  function grad_compare(obj1, obj2) {
    return obj2.grad - obj1.grad
  }

  // sort and take top-k
  const topKPremiseGrads = premiseGradsWithIdx.sort(grad_compare).slice(0, premiseTopK)
  console.log('TOPK PREMISE GRADS', topKPremiseGrads)
  const topKHypothesisGrads = hypothesisGradsWithIdx.sort(grad_compare).slice(0, hypothesisTopK)

  // Store set of weights we want to visualize
  const validPremiseGrads = new Set(topKPremiseGrads.map((el) => el.idx))
  const validHypothesisGrads = new Set(topKHypothesisGrads.map((el) => el.idx))

  // We do 1 - weight to get the colormap scaling right
  const premiseTokensWithWeights = premise_tokens.map((token, idx) => {
    if (validPremiseGrads.has(idx)) {
      let weight = premiseGrads[idx]
      return {token, weight: 1 - weight}
    } else {
      return {token, weight: undefined}
    }
  })

  // We do 1 - weight to get the colormap scaling right
  const hypothesisTokensWithWeights = hypothesis_tokens.map((token, idx) => {
    if (validHypothesisGrads.has(idx)) {
      let weight = hypothesisGrads[idx]
      return {token, weight: 1 - weight}
    } else {
      return {token, weight: undefined}
    }
  })

  return [premiseTokensWithWeights, hypothesisTokensWithWeights]
}

class Output extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      integratedGradients: {
        premiseTopK: 3,
        hypothesisTopK: 3
      },
      simpleGradients: {
        premiseTopK: 3,
        hypothesisTopK: 3 
      }
    }

    this.handleTopKChange = (fieldName, interpreter) => e => {
      if (e.target.value.trim() !== "") {
        let stateUpdate = Object.assign({}, this.state)
        console.log('state update', stateUpdate)
        stateUpdate[interpreter][fieldName] = parseInt(e.target.value, 10)
        this.setState(stateUpdate)
      }
    }
  }

  render() {
    const { requestData, responseData, interpretData, interpretModel } = this.props 
    const { label_probs, h2p_attention, p2h_attention, premise_tokens, hypothesis_tokens } = responseData

    const { simple_gradients_interpreter, integrated_gradients_interpreter } = interpretData ? interpretData : {'simple_gradients_interpreter': undefined, 'integrated_gradients_interpreter': undefined} 
  
    let gradientPremiseTokensWithWeights = []
    let gradientHypothesisTokensWithWeights = []

    let igPremiseTokensWithWeights = []
    let igHypothesisTokensWithWeights = []

    if (simple_gradients_interpreter) {
      const { instance_1 } = simple_gradients_interpreter
      const { grad_input_1, grad_input_2 } = instance_1 

      const tokensWithWeights = getTokenWeightPairs(grad_input_2, grad_input_1, premise_tokens, hypothesis_tokens, this.state.simpleGradients.premiseTopK, this.state.simpleGradients.hypothesisTopK)
      gradientPremiseTokensWithWeights = tokensWithWeights[0]
      gradientHypothesisTokensWithWeights = tokensWithWeights[1]
    }

    if (integrated_gradients_interpreter) {
      const { instance_1 } = integrated_gradients_interpreter
      const { grad_input_1, grad_input_2 } = instance_1 

      const tokensWithWeights = getTokenWeightPairs(grad_input_2, grad_input_1, premise_tokens, hypothesis_tokens, this.state.integratedGradients.premiseTopK, this.state.integratedGradients.hypothesisTopK)
      igPremiseTokensWithWeights = tokensWithWeights[0]
      igHypothesisTokensWithWeights = tokensWithWeights[1]
    }


    const [entailment, contradiction, neutral] = label_probs
    
    // request data contains {premise: "...", hypothesis: "..."} just as we would expect 

    // Find judgment and confidence.
    let judgment
    let confidence

    if (entailment > contradiction && entailment > neutral) {
      judgment = judgments.ENTAILMENT
      confidence = entailment
    }
    else if (contradiction > entailment && contradiction > neutral) {
      judgment = judgments.CONTRADICTION
      confidence = contradiction
    }
    else if (neutral > entailment && neutral > contradiction) {
      judgment = judgments.NEUTRAL
      confidence = neutral
    } else {
      throw new Error("cannot form judgment")
    }

    // Create summary text.
    const veryConfident = 0.75;
    const somewhatConfident = 0.50;
    let summaryText

    if (confidence >= veryConfident) {
      summaryText = (
        <div>
          It is <strong>very likely</strong> that {judgment}.
        </div>
      )
    } else if (confidence >= somewhatConfident) {
      summaryText = (
        <div>
          It is <strong>somewhat likely</strong> that {judgment}.
        </div>
      )
    } else {
      summaryText = (
        <div>The model is not confident in its judgment.</div>
        )
    }

    function formatProb(n) {
      return parseFloat((n * 100).toFixed(1)) + "%";
    }

    // https://en.wikipedia.org/wiki/Ternary_plot#Plotting_a_ternary_plot
    const a = contradiction;
    const b = neutral;
    const c = entailment;
    const x = 0.5 * (2 * b + c) / (a + b + c)
    const y = (c / (a + b + c))

    return (
    <div className="model__content answer">
      <OutputField label="Summary">
      {summaryText}
      </OutputField>
      <div className="te-output">
      <TeGraph x={x} y={y}/>
      <div className="te-table">
        <table>
        <thead>
          <tr>
          <th>Judgment</th>
          <th>Probability</th>
          </tr>
        </thead>
        <tbody>
          <tr>
          <td>Entailment</td>
          <td>{formatProb(entailment)}</td>
          </tr>
          <tr>
          <td>Contradiction</td>
          <td>{formatProb(contradiction)}</td>
          </tr>
          <tr>
          <td>Neutral</td>
          <td>{formatProb(neutral)}</td>
          </tr>
        </tbody>
        </table>
      </div>
      </div>
      <OutputField label=" Model internals">
        <Accordion accordion={false}>

          {/* ************************** */}
          {/* SIMPLE GRADIENTS COMPONENT */}
          {/* ************************** */}
          <AccordionItem expanded={true}>
            <AccordionItemTitle>
              Simple Gradients Interpretation Visualization
              <div className="accordion__arrow" role="presentation"/>
            </AccordionItemTitle>
            <AccordionItemBody>
              <br />
              <strong>Top K gradients (premise):</strong> <input type="text" value={this.state.simpleGradients.premiseTopK} onChange={this.handleTopKChange('premiseTopK', 'simpleGradients')}/><br />
              <strong>Top K gradients (hypothesis):</strong> <input type="text" value={this.state.simpleGradients.hypothesisTopK} onChange={this.handleTopKChange('hypothesisTopK', 'simpleGradients')}/> 
              <br />
              <p><strong>Saliency Map for Premise:</strong></p>
              {gradientPremiseTokensWithWeights.length !== 0 ? <TextSaliencyMap tokensWithWeights={gradientPremiseTokensWithWeights} colormapProps={{colormap: 'copper',
                                                                                            format: 'hex',
                                                                                            nshades: 20}} /> : <p style={{color: "#7c7c7c"}}>Press "interpret prediction" to show premise interpretation</p>}
                                                                                           
              <p><strong>Saliency Map for Hypothesis:</strong></p>
              {gradientHypothesisTokensWithWeights.length !== 0 ? <TextSaliencyMap tokensWithWeights={gradientHypothesisTokensWithWeights} colormapProps={{colormap: 'copper',
                                                                                              format: 'hex',
                                                                                              nshades: 20}} /> : <p style={{color: "#7c7c7c"}}>Press "interpret prediction" to show hypothesis interpretation</p>}
                                                                                              
              <button
                type="button"
                className="btn"
                style={{margin: "30px 0px"}}
                onClick={ () => interpretModel(requestData, 'simple_gradients_interpreter') }>Interpret Prediction
              </button>
            </AccordionItemBody>
          </AccordionItem>

          {/* ****************************** */}
          {/* INTEGRATED GRADIENTS COMPONENT */}
          {/* ****************************** */}
          <AccordionItem expanded={true}>
            <AccordionItemTitle>
              Integrated Gradients Interpretation Visualization
              <div className="accordion__arrow" role="presentation"/>
            </AccordionItemTitle>
            <AccordionItemBody>
              <br />
              <strong>Top K gradients (premise):</strong> <input type="text" value={this.state.integratedGradients.premiseTopK} onChange={this.handleTopKChange('premiseTopK', 'integratedGradients')}/><br />
              <strong>Top K gradients (hypothesis):</strong> <input type="text" value={this.state.integratedGradients.hypothesisTopK} onChange={this.handleTopKChange('hypothesisTopK', 'integratedGradients')}/> 
              <br />
              <p><strong>Saliency Map for Premise:</strong></p>
              {igPremiseTokensWithWeights.length !== 0 ? <TextSaliencyMap tokensWithWeights={igPremiseTokensWithWeights} colormapProps={{colormap: 'copper',
                                                                                                                                    format: 'hex',
                                                                                                                                    nshades: 20}} /> : <p style={{color: "#7c7c7c"}}>Press "interpret prediction" to show premise interpretation</p>}
                                                                                           
              <p><strong>Saliency Map for Hypothesis:</strong></p>
              {igHypothesisTokensWithWeights.length !== 0 ? <TextSaliencyMap tokensWithWeights={igHypothesisTokensWithWeights} colormapProps={{colormap: 'copper',
                                                                                                                                          format: 'hex',
                                                                                                                                          nshades: 20}} /> : <p style={{color: "#7c7c7c"}}>Press "interpret prediction" to show hypothesis interpretation</p>}
                                                                                              
              <button
                type="button"
                className="btn"
                style={{margin: "30px 0px"}}
                onClick={ () => interpretModel(requestData, 'integrated_gradients_interpreter') }>Interpret Prediction
              </button>
            </AccordionItemBody>
          </AccordionItem>
          <AccordionItem expanded={true}>
            <AccordionItemTitle>
              Premise to Hypothesis Attention
              <div className="accordion__arrow" role="presentation"/>
            </AccordionItemTitle>
            <AccordionItemBody>
              <p>
                  For every premise word, the model computes an attention over the hypothesis words.
                  This heatmap shows that attention, which is normalized for every row in the matrix.
              </p>
              <HeatMap colLabels={premise_tokens} rowLabels={hypothesis_tokens} data={h2p_attention} />
            </AccordionItemBody>
          </AccordionItem>
          <AccordionItem>
            <AccordionItemTitle>
              Hypothesis to Premise Attention
              <div className="accordion__arrow" role="presentation"/>
            </AccordionItemTitle>
            <AccordionItemBody>
              <p>
                For every hypothesis word, the model computes an attention over the premise words.
                This heatmap shows that attention, which is normalized for every row in the matrix.
              </p>
              <HeatMap colLabels={hypothesis_tokens} rowLabels={premise_tokens} data={p2h_attention} />
            </AccordionItemBody>
          </AccordionItem>
        </Accordion>
      </OutputField>
    </div>
    );
  }
}

const examples = [
  {
    premise: "If you help the needy, God will reward you.",
    hypothesis: "Giving money to the poor has good consequences.",
  },
  {
    premise: "Two women are wandering along the shore drinking iced tea.",
    hypothesis: "Two women are sitting on a blanket near some rocks talking about politics.",
  },
  {
    premise: "An interplanetary spacecraft is in orbit around a gas giant's icy moon.",
    hypothesis: "The spacecraft has the ability to travel between planets.",
  },
  {
    premise: "A large, gray elephant walked beside a herd of zebras.",
    hypothesis: "The elephant was lost.",
  },
  {
    premise: "A handmade djembe was on display at the Smithsonian.",
    hypothesis: "Visitors could see the djembe.",
  },
]

const modelProps = {apiUrl, apiUrlInterpret, title, description, descriptionEllipsed, fields, examples, Output}

// withRouter will pass updated match, location, and history props to the wrapped
// component 
export default withRouter(props => <Model {...props} {...modelProps}/>)
