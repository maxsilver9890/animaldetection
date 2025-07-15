import React, { useState, useCallback, useEffect } from 'react'; // FIX: Added useEffect
import * as tf from '@tensorflow/tfjs';
import { ModelContext } from './detection/context/model-context';
import Selector from './detection/utils/Selector';
import LoadingSpinner from './detection/utils/LoadingSpinner';

// FIX: Replace the local path with the absolute URL to your hosted model
const MODEL_URL = 'https://github.com/maxsilver9890/animaldetection/releases/download/v1.0.0-model/model.json';
const LABELS_URL = 'https://raw.githubusercontent.com/maxsilver9890/animaldetection/refs/tags/v1.0.0-model/WildEye/public/detection/labels.json'; // FIX: Also use the absolute URL for labels

const Detect = () => {
    const [model, setModel] = useState(null);
    const [labels, setLabels] = useState(null);
    const [loading, setLoading] = useState(false);
    const [selected, setSelected] = useState('');

    const fetchModel = useCallback((model) => {
        setModel(model);
    }, []);

    const fetchLabels = useCallback((labels) => {
        setLabels(labels);
    }, []);

    const selectMode = useCallback((selected) => {
        setSelected(selected)
    }, [])

    // FIX: Changed to a useEffect hook to load the model automatically on component mount
    useEffect(() => {
        const loadModel = async () => {
            setLoading(true);
            try {
                const savedModel = await tf.io.listModels();
                if (savedModel['indexeddb://animal_detector']) {
                    // If model is found in IndexedDB, load it
                    const model = await tf.loadGraphModel('indexeddb://animal_detector');
                    const response = await fetch(LABELS_URL); // Still fetch labels from remote
                    let labels_json = await response.json();
                    
                    fetchModel(model);
                    fetchLabels(labels_json);
                    console.log("Loaded model from browser cache.");
                } else {
                    // If not in cache, load from remote URL for the first time
                    console.log("Loading model from remote for the first time...");
                    const model = await tf.loadGraphModel(MODEL_URL);
                    fetchModel(model);

                    const response = await fetch(LABELS_URL);
                    let labels_json = await response.json();
                    fetchLabels(labels_json);
                    
                    // Save the model to IndexedDB for future visits
                    await model.save('indexeddb://animal_detector');
                    console.log("Model saved to browser cache.");
                }
            } catch (error) {
                console.error("Failed to load or process model:", error);
                // Optionally, set an error state here to show a message to the user
            }
            setLoading(false);
        };

        loadModel();
    }, [fetchModel, fetchLabels]); // The empty dependency array ensures this runs only once

    return (
        <ModelContext.Provider
            value={{
                model: model,
                fetchModel: fetchModel,
                labels: labels,
                fetchLabels: fetchLabels,
                selected: selected,
                selectMode: selectMode
            }}>
            <div className="header-div">
                <p className="demo-title">Object Detection</p>
                <p>This currently uses YOLOv11n pretrained coco model</p>
            </div>
            <div>
                {model ? (
                    <div>
                        <Selector />
                    </div>
                ) : (
                    <div className="center-div load-div">
                        {loading ? (
                            <div style={{ textAlign: 'center' }}>
                                <LoadingSpinner />
                                <p style={{
                                    color: '#950740',
                                    fontWeight: '500',
                                }}>Loading Model. This may take a moment on the first visit...</p>
                            </div>
                        ) : (
                            // This part might not be shown if loading starts automatically,
                            // but it's good as a fallback.
                            <p>Failed to load model. Please try refreshing the page.</p>
                        )}
                    </div>
                )}
            </div>
        </ModelContext.Provider>
    );
}

export default Detect;
