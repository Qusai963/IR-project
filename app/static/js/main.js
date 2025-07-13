const API_BASE_URL = "/api";
const ENDPOINTS = {
  SEARCH: `${API_BASE_URL}/search`,
  QUERIES: `${API_BASE_URL}/queries`,
  DOCS: `${API_BASE_URL}/docs`,
  QRELS: `${API_BASE_URL}/qrels`,
  EXPORT: `${API_BASE_URL}/export`,
  QUERY_REFINEMENT: `${API_BASE_URL}/query-refinement`,
  CLUSTERING: `${API_BASE_URL}/clustering`,
  TOPIC_MODELING: `${API_BASE_URL}/topic-modeling`,
  HYBRID_SEARCH: `${API_BASE_URL}/hybrid/search`,
  RAG: `${API_BASE_URL}/rag`,
};

const searchForm = document.getElementById("searchForm");
const datasetSelect = document.getElementById("datasetSelect");
const modelSelect = document.getElementById("modelSelect");
const queryInput = document.getElementById("queryInput");
const topKInput = document.getElementById("topK");
const resultsContainer = document.getElementById("resultsContainer");
const loadingSpinner = document.getElementById("loadingSpinner");
const suggestionsPanel = document.getElementById("suggestionsPanel");
const expandedQueriesList = document.querySelector(
  "#expandedQueries .suggestion-list"
);
const queryRefinementCheckbox = document.getElementById("queryRefinement");
const clearQueryButton = document.getElementById("clearQuery");
const closeSuggestionsButton = document.getElementById("closeSuggestions");

// RAG Tab elements
const ragForm = document.getElementById("ragForm");
const ragDatasetSelect = document.getElementById("ragDatasetSelect");
const ragQueryInput = document.getElementById("ragQueryInput");
const ragTopKInput = document.getElementById("ragTopK");
const createVectorStoreBtn = document.getElementById("createVectorStoreBtn");
const ragResultsContainer = document.getElementById("ragResultsContainer");
const ragLoadingSpinner = document.getElementById("ragLoadingSpinner");
const ragStatus = document.getElementById("ragStatus");

// Advanced features elements
const advancedFeaturesBtn = document.getElementById("advancedFeaturesBtn");
const hybridOptions = document.getElementById("hybridOptions");
const tfidfWeight = document.getElementById("tfidfWeight");
const bertWeight = document.getElementById("bertWeight");
const bm25Weight = document.getElementById("bm25Weight");

const datasets = [
  { id: "antique", name: "Antique" },
  { id: "quora", name: "Quora" },
];

// Preprocess Tab Logic
const preprocessForm = document.getElementById("preprocessForm");
const preprocessText = document.getElementById("preprocessText");
const preprocessResult = document.getElementById("preprocessResult");

function initializeUI() {
  datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = dataset.name;
    datasetSelect.appendChild(option);

    // Also populate RAG dataset select
    const ragOption = document.createElement("option");
    ragOption.value = dataset.id;
    ragOption.textContent = dataset.name;
    ragDatasetSelect.appendChild(ragOption);
  });

  searchForm.addEventListener("submit", handleSearch);
  queryInput.addEventListener("input", handleQueryInput);
  queryRefinementCheckbox.addEventListener(
    "change",
    handleQueryRefinementToggle
  );
  clearQueryButton.addEventListener("click", clearQuery);
  closeSuggestionsButton.addEventListener("click", hideSuggestions);

  // RAG Tab functionality
  ragForm.addEventListener("submit", handleRAGFormSubmit);
  createVectorStoreBtn.addEventListener("click", handleCreateVectorStore);
  ragDatasetSelect.addEventListener("change", handleRAGDatasetChange);

  // Advanced features
  advancedFeaturesBtn.addEventListener("click", showAdvancedFeatures);

  // Ensure model change handler is properly attached
  if (modelSelect) {
    // Remove any existing listeners first
    modelSelect.removeEventListener("change", handleModelChange);
    // Add the listener
    modelSelect.addEventListener("change", handleModelChange);
    // Also trigger on load to check current state
    setTimeout(() => handleModelChange(), 100);
  }

  document.addEventListener("click", (event) => {
    if (
      !suggestionsPanel.contains(event.target) &&
      event.target !== queryInput
    ) {
      hideSuggestions();
    }
  });

  if (preprocessForm) {
    preprocessForm.addEventListener("submit", async function (e) {
      e.preventDefault();
      preprocessResult.innerHTML = "";
      const text = preprocessText.value;
      try {
        const response = await axios.get(`/api/preprocess/any`, {
          params: { text },
        });
        const data = response.data;
        if (Array.isArray(data) && data.length > 0) {
          const item = data[0];
          preprocessResult.innerHTML = `
            <div class="card">
              <div class="card-body">
                <h6 class="card-title">Original Text</h6>
                <p class="card-text"><code>${item.original}</code></p>
                <h6 class="card-title mt-3">Processed Text</h6>
                <p class="card-text text-success"><code>${item.processed}</code></p>
              </div>
            </div>
          `;
        } else {
          preprocessResult.innerHTML = `<div class="alert alert-warning">No result returned.</div>`;
        }
      } catch (error) {
        preprocessResult.innerHTML = `<div class="alert alert-danger">Error: ${
          error?.response?.data?.detail || error.message
        }</div>`;
      }
    });
  }

  // Tab switching for main tabs (Search/RAG/Preprocess)
  document
    .getElementById("searchTabBtn")
    .addEventListener("click", function (e) {
      e.preventDefault();
      document.getElementById("searchTab").classList.add("show", "active");
      document.getElementById("ragTab").classList.remove("show", "active");
      document
        .getElementById("preprocessTab")
        .classList.remove("show", "active");
      this.classList.add("active");
      document.getElementById("ragTabBtn").classList.remove("active");
      document.getElementById("preprocessTabBtn").classList.remove("active");
    });
  document.getElementById("ragTabBtn").addEventListener("click", function (e) {
    e.preventDefault();
    document.getElementById("ragTab").classList.add("show", "active");
    document.getElementById("searchTab").classList.remove("show", "active");
    document.getElementById("preprocessTab").classList.remove("show", "active");
    this.classList.add("active");
    document.getElementById("searchTabBtn").classList.remove("active");
    document.getElementById("preprocessTabBtn").classList.remove("active");
  });
  document
    .getElementById("preprocessTabBtn")
    .addEventListener("click", function (e) {
      e.preventDefault();
      document.getElementById("preprocessTab").classList.add("show", "active");
      document.getElementById("searchTab").classList.remove("show", "active");
      document.getElementById("ragTab").classList.remove("show", "active");
      this.classList.add("active");
      document.getElementById("searchTabBtn").classList.remove("active");
      document.getElementById("ragTabBtn").classList.remove("active");
    });
}

function handleModelChange() {
  const selectedModel = modelSelect.value;

  if (selectedModel.includes("hybrid")) {
    if (hybridOptions) {
      hybridOptions.style.display = "block";
    }
    // Add visual feedback
    modelSelect.style.borderColor = "#007bff";
    modelSelect.style.boxShadow = "0 0 0 0.2rem rgba(0, 123, 255, 0.25)";
  } else {
    if (hybridOptions) {
      hybridOptions.style.display = "none";
    }
    // Remove visual feedback
    modelSelect.style.borderColor = "";
    modelSelect.style.boxShadow = "";
  }
}

// Global function for setting weights (called from HTML buttons)
function setWeights(weights) {
  if (weights.length >= 1) tfidfWeight.value = weights[0];
  if (weights.length >= 2) bertWeight.value = weights[1];
  if (weights.length >= 3) bm25Weight.value = weights[2];
}

function showAdvancedFeatures() {
  const modal = new bootstrap.Modal(
    document.getElementById("advancedFeaturesModal")
  );
  modal.show();

  // Initialize advanced features event listeners
  initializeAdvancedFeatures();
}

function initializeAdvancedFeatures() {
  // Clustering
  document
    .getElementById("fitClustering")
    .addEventListener("click", handleFitClustering);

  // View Clustering Results
  document
    .getElementById("viewClusteringResults")
    .addEventListener("click", handleViewClusteringResults);

  // Topic modeling
  document
    .getElementById("fitTopicModeling")
    .addEventListener("click", handleFitTopicModeling);
}

async function handleSearch(event) {
  event.preventDefault();

  const searchData = {
    dataset_name: datasetSelect.value,
    query: queryInput.value,
    model: modelSelect.value,
    top_k: parseInt(topKInput.value),
  };

  // Add hybrid-specific parameters
  if (modelSelect.value.includes("hybrid")) {
    searchData.weights = [
      parseFloat(tfidfWeight.value),
      parseFloat(bertWeight.value),
      parseFloat(bm25Weight.value),
    ];
  }

  if (queryRefinementCheckbox.checked) {
    searchData.query_refinement = true;
  }

  try {
    showLoading(true);

    // Use the main search endpoint for all models (including hybrid)
    const endpoint = ENDPOINTS.SEARCH;

    const response = await axios.post(endpoint, searchData);
    displayResults(response.data);
    hideSuggestions();
  } catch (error) {
    showError("Error performing search: " + error.message);
  } finally {
    showLoading(false);
  }
}

function displayResults(data) {
  resultsContainer.innerHTML = "";

  if (!data.results || data.results.length === 0) {
    resultsContainer.innerHTML =
      '<div class="text-center text-muted">No results found</div>';
    return;
  }

  // Add model info header for hybrid search
  let headerHtml = "";
  if (data.model && data.model.includes("hybrid")) {
    const weights = data.weights || [];
    const weightLabels = ["TF-IDF", "BERT", "BM25"];
    const activeModels = weights
      .map((w, i) => (w > 0 ? weightLabels[i] : null))
      .filter(Boolean);

    headerHtml = `
      <div class="alert alert-info mb-3">
        <strong>Hybrid Search Configuration:</strong><br>
        <small>Active models: ${activeModels.join(", ")}<br>
        Weights: ${weights
          .map((w, i) => `${weightLabels[i]}: ${w.toFixed(2)}`)
          .join(" | ")}<br>
        Results: ${data.results.length} documents found</small>
      </div>
    `;
  } else {
    // Show results count for non-hybrid searches
    headerHtml = `
      <div class="alert alert-success mb-3">
        <strong>Search Results:</strong> ${data.results.length} documents found
      </div>
    `;
  }

  const resultsList = document.createElement("div");
  resultsList.className = "list-group";

  data.results.forEach((result) => {
    const resultItem = document.createElement("div");
    resultItem.className = "result-item";
    resultItem.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <h6 class="mb-1">Document ID: ${result.doc_id}</h6>
                <span class="result-score">Score: ${result.score.toFixed(
                  4
                )}</span>
            </div>
            <p class="result-text mb-0">${result.text}</p>
        `;
    resultsList.appendChild(resultItem);
  });

  if (headerHtml) {
    resultsContainer.innerHTML = headerHtml;
  }
  resultsContainer.appendChild(resultsList);
}

async function handleQueryInput(event) {
  const query = event.target.value.trim();

  if (query.length < 3) {
    hideSuggestions();
    return;
  }

  if (queryRefinementCheckbox.checked) {
    try {
      const response = await axios.post(ENDPOINTS.QUERY_REFINEMENT, { query });
      displaySuggestions(response.data);
    } catch (error) {
      console.error("Error getting suggestions:", error);
    }
  }
}

function displaySuggestions(data) {
  if (!data || !data.expanded_queries) {
    hideSuggestions();
    return;
  }

  expandedQueriesList.innerHTML = "";

  if (data.expanded_queries && data.expanded_queries.length > 0) {
    data.expanded_queries.forEach((query) => {
      const suggestionItem = createSuggestionItem(query, "fa-sync-alt");
      expandedQueriesList.appendChild(suggestionItem);
    });
  }

  showSuggestions();
}

function createSuggestionItem(query, iconClass) {
  const suggestionItem = document.createElement("div");
  suggestionItem.className = "suggestion-item";
  suggestionItem.innerHTML = `
        <i class="fas ${iconClass}"></i>
        <span>${query}</span>
    `;
  suggestionItem.addEventListener("click", () => {
    queryInput.value = query;
    hideSuggestions();
  });
  return suggestionItem;
}

function showSuggestions() {
  suggestionsPanel.classList.add("show");
}

function hideSuggestions() {
  suggestionsPanel.classList.remove("show");
}

function clearQuery() {
  queryInput.value = "";
  hideSuggestions();
  queryInput.focus();
}

function handleQueryRefinementToggle(event) {
  if (!event.target.checked) {
    hideSuggestions();
  }
}

// Advanced Features Handlers
async function handleFitClustering() {
  const dataset = document.getElementById("clusteringDataset").value;
  const k = document.getElementById("clusteringK").value;

  const requestData = {
    dataset_name: dataset,
    save_visualization: true,
  };

  if (k) {
    requestData.k = parseInt(k);
  }

  try {
    showLoading(true);
    const response = await axios.post(
      ENDPOINTS.CLUSTERING + "/fit",
      requestData
    );

    const resultsDiv = document.getElementById("clusteringResults");
    let html = `
      <div class="alert alert-success">
        <h6>Clustering Results:</h6>
        <p><strong>Optimal K:</strong> ${response.data.optimal_k}</p>
        <p><strong>Silhouette Score:</strong> ${response.data.silhouette_score.toFixed(
          4
        )}</p>
        <p><strong>Cluster Distribution:</strong></p>
        <ul>
          ${Object.entries(response.data.cluster_distribution)
            .map(
              ([cluster, count]) =>
                `<li>Cluster ${cluster}: ${count} documents</li>`
            )
            .join("")}
        </ul>
      </div>
    `;
    // Add visualization image if available
    html += `<div class="mt-3"><h6>Cluster Visualization</h6><img src="/api/clustering/${dataset}/visualization?${Date.now()}" alt="Clustering Visualization" style="max-width:100%;border:1px solid #ccc;" onerror="this.style.display='none'" /></div>`;
    resultsDiv.innerHTML = html;
  } catch (error) {
    showError("Error fitting clustering model: " + error.message);
  } finally {
    showLoading(false);
  }
}

async function handleFitTopicModeling() {
  const dataset = document.getElementById("topicDataset").value;
  const n = document.getElementById("topicN").value;

  const requestData = {
    dataset_name: dataset,
    save_visualization: true,
  };

  if (n) {
    requestData.n_topics = parseInt(n);
  }

  try {
    showLoading(true);
    const response = await axios.post(
      ENDPOINTS.TOPIC_MODELING + "/fit",
      requestData
    );

    const resultsDiv = document.getElementById("topicModelingResults");
    let topicsHtml = "";

    Object.entries(response.data.topic_words).forEach(([topicId, words]) => {
      topicsHtml += `
        <div class="mb-2">
          <strong>Topic ${parseInt(topicId) + 1}:</strong>
          <span>${words.map((w) => w.word).join(", ")}</span>
        </div>
      `;
    });

    resultsDiv.innerHTML = `
      <div class="alert alert-success">
        <h6>Topic Modeling Results:</h6>
        <p><strong>Optimal Topics:</strong> ${response.data.optimal_topics}</p>
        <p><strong>Perplexity:</strong> ${response.data.perplexity.toFixed(
          2
        )}</p>
        <p><strong>Log Likelihood:</strong> ${response.data.log_likelihood.toFixed(
          2
        )}</p>
        <p><strong>Top Words by Topic:</strong></p>
        ${topicsHtml}
      </div>
    `;
  } catch (error) {
    showError("Error fitting topic model: " + error.message);
  } finally {
    showLoading(false);
  }
}

// New RAG Tab Functions
async function handleRAGFormSubmit(event) {
  event.preventDefault();

  const query = ragQueryInput.value.trim();
  const topK = parseInt(ragTopKInput.value) || 3;

  if (!query) {
    showRAGError("Please enter a question");
    return;
  }

  try {
    showRAGLoading(true);
    const response = await axios.post(ENDPOINTS.RAG + "/search", {
      query: query,
      top_k: topK,
    });

    displayRAGResults(response.data);
  } catch (error) {
    showRAGError(
      "Error generating answer: " +
        (error?.response?.data?.detail || error.message)
    );
  } finally {
    showRAGLoading(false);
  }
}

async function handleCreateVectorStore() {
  const dataset = ragDatasetSelect.value;

  if (!dataset) {
    showRAGError("Please select a dataset first");
    return;
  }

  try {
    showRAGLoading(true);
    updateRAGStatus("Initializing RAG from BERT embeddings...", "info");

    const response = await axios.post(ENDPOINTS.RAG + "/initialize", null, {
      params: { dataset_name: dataset },
    });

    updateRAGStatus("RAG initialized successfully!", "success");
    displayRAGVectorStoreInfo(response.data);
  } catch (error) {
    showRAGError(
      "Error initializing RAG: " +
        (error?.response?.data?.detail || error.message)
    );
    updateRAGStatus("Failed to initialize RAG", "danger");
  } finally {
    showRAGLoading(false);
  }
}

function handleRAGDatasetChange() {
  const dataset = ragDatasetSelect.value;
  if (dataset) {
    updateRAGStatus(`Dataset selected: ${dataset}`, "info");
  } else {
    updateRAGStatus("Select a dataset and initialize RAG to start", "muted");
  }
}

function displayRAGResults(data) {
  let documentsHtml = "";

  if (data.retrieved_documents && data.retrieved_documents.length > 0) {
    data.retrieved_documents.forEach((doc, index) => {
      documentsHtml += `
        <div class="mb-3 p-3 border rounded bg-light">
          <div class="d-flex justify-content-between align-items-start mb-2">
            <strong class="text-primary">Document ${index + 1}</strong>
            <span class="badge bg-secondary">Score: ${doc.score.toFixed(
              4
            )}</span>
          </div>
          <p class="mb-0">${doc.text}</p>
        </div>
      `;
    });
  }

  // Determine generation method badge
  const generationMethod = data.generation_method || "template";
  const methodBadge =
    generationMethod === "llm"
      ? '<span class="badge bg-success">AI Generated</span>'
      : '<span class="badge bg-warning">Template Based</span>';

  ragResultsContainer.innerHTML = `
    <div class="alert alert-success">
      <div class="d-flex justify-content-between align-items-start mb-2">
        <h5 class="alert-heading mb-0">
          <i class="fas fa-robot me-2"></i>Generated Answer
        </h5>
        ${methodBadge}
      </div>
      <p class="mb-3">${data.answer}</p>
      <hr>
      <h6 class="mb-2">
        <i class="fas fa-file-alt me-2"></i>Retrieved Context 
        <span class="badge bg-info">${
          data.context_used || data.retrieved_documents?.length || 0
        } documents used</span>
      </h6>
      ${documentsHtml}
    </div>
  `;
}

function displayRAGVectorStoreInfo(data) {
  // Check if LLM is available
  const llmStatus = data.llm_available
    ? '<span class="badge bg-success">AI Model Available</span>'
    : '<span class="badge bg-warning">Template Mode</span>';

  // Check embedding source
  const embeddingSource =
    data.details.embeddings_source === "existing_bert_model"
      ? '<span class="badge bg-primary">BERT Embeddings</span>'
      : '<span class="badge bg-secondary">Custom Embeddings</span>';

  ragResultsContainer.innerHTML = `
    <div class="alert alert-info">
      <div class="d-flex justify-content-between align-items-start mb-2">
        <h5 class="alert-heading mb-0">
          <i class="fas fa-robot me-2"></i>RAG Initialized
        </h5>
        <div>
          ${llmStatus}
          ${embeddingSource}
        </div>
      </div>
      <p class="mb-2">${data.message}</p>
      <div class="row">
        <div class="col-md-6">
          <strong>Documents:</strong> ${data.details.num_documents}
        </div>
        <div class="col-md-6">
          <strong>BERT Model:</strong> ${data.details.bert_model_used}
        </div>
      </div>
      ${
        data.details.dataset_name
          ? `<div class="mt-2"><strong>Dataset:</strong> ${data.details.dataset_name}</div>`
          : ""
      }
      <hr>
      <p class="mb-0 text-muted">
        <i class="fas fa-info-circle me-1"></i>
        RAG is now ready! You can ask questions and get ${
          data.llm_available ? "AI-generated" : "template-based"
        } answers based on this dataset.
        Uses existing BERT embeddings for optimal performance.
      </p>
    </div>
  `;
}

function updateRAGStatus(message, type = "muted") {
  const iconClass =
    type === "success"
      ? "fa-check-circle"
      : type === "danger"
      ? "fa-exclamation-circle"
      : type === "info"
      ? "fa-info-circle"
      : "fa-info-circle";

  ragStatus.className = `text-${type}`;
  ragStatus.innerHTML = `<i class="fas ${iconClass} me-2"></i>${message}`;
}

function showRAGLoading(show) {
  ragLoadingSpinner.classList.toggle("d-none", !show);
}

function showRAGError(message) {
  ragResultsContainer.innerHTML = `
    <div class="alert alert-danger alert-dismissible fade show">
      <i class="fas fa-exclamation-triangle me-2"></i>
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;
}

async function handleViewClusteringResults() {
  const dataset = document.getElementById("clusteringDataset").value;
  const resultsDiv = document.getElementById("clusteringResults");
  resultsDiv.innerHTML =
    "<span class='text-muted'>Loading clustering summary...</span>";
  try {
    const response = await axios.get(`/api/clustering/${dataset}/summary`);
    let html = `<div class='alert alert-info'>`;
    html += `<h6>Clustering Summary:</h6>`;
    if (typeof response.data === "object") {
      html += `<pre>${JSON.stringify(response.data, null, 2)}</pre>`;
    } else {
      html += `<p>${response.data}</p>`;
    }
    html += `</div>`;
    // Add visualization image
    html += `<div class='mt-3'><h6>Cluster Visualization</h6><img src="/api/clustering/${dataset}/visualization?${Date.now()}" alt="Clustering Visualization" style="max-width:100%;border:1px solid #ccc;" onerror="this.style.display='none'" /></div>`;
    resultsDiv.innerHTML = html;
  } catch (error) {
    resultsDiv.innerHTML = `<div class='alert alert-danger'>Error loading clustering summary: ${error.message}</div>`;
  }
}

function showLoading(show) {
  loadingSpinner.classList.toggle("d-none", !show);
}

function showError(message) {
  const alertDiv = document.createElement("div");
  alertDiv.className = "alert alert-danger alert-dismissible fade show";
  alertDiv.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
  resultsContainer.appendChild(alertDiv);
}

// Initialize the application
document.addEventListener("DOMContentLoaded", initializeUI);
