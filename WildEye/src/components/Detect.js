import React, { useState, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { ModelContext } from './detection/context/model-context';
import Selector from './detection/utils/Selector';
import LoadingSpinner from './detection/utils/LoadingSpinner';

// FIX: Use jsDelivr CDN to serve files with correct CORS headers
const MODEL_URL = 'https://cdn.jsdelivr.net/gh/maxsilver9890/animaldetection@main/public/detection/model.json';
const LABELS_URL = 'https://cdn.jsdelivr.net/gh/maxsilver9890/animaldetection@main/public/detection/labels.json';

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

    useEffect(() => {
        const loadModel = async () => {
            setLoading(true);
            try {
                const savedModel = await tf.io.listModels();
                if (savedModel['indexeddb://animal_detector']) {
                    const model = await tf.loadGraphModel('indexeddb://animal_detector');
                    const response = await fetch(LABELS_URL);
                    let labels_json = await response.json();
                    
                    fetchModel(model);
                    fetchLabels(labels_json);
                    console.log("Loaded model from browser cache.");
                } else {
                    console.log("Loading model from remote for the first time via jsDelivr...");
                    const model = await tf.loadGraphModel(MODEL_URL);
                    fetchModel(model);

                    const response = await fetch(LABELS_URL);
                    let labels_json = await response.json();
                    fetchLabels(labels_json);
                    
                    await model.save('indexeddb://animal_detector');
                    console.log("Model saved to browser cache.");
                }
            } catch (error) {
                console.error("Failed to load or process model:", error);
            }
            setLoading(false);
        };

        loadModel();
    }, [fetchModel, fetchLabels]);

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
                            <p>Failed to load model. Please try refreshing the page.</p>
                        )}
                    </div>
                )}
            </div>
        </ModelContext.Provider>
    );
}

export default Detect;
