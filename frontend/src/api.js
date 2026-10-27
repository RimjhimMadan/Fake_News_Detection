export async function readJsonResponse(response) {
  const body = await response.text();
  let data = {};
  let parsed = false;

  if (body.trim()) {
    try {
      data = JSON.parse(body);
      parsed = true;
    } catch {
      data = {};
    }
  }

  if (!response.ok) {
    throw new Error(data.detail || `Server returned error ${response.status}`);
  }

  if (!body.trim()) {
    throw new Error('The server returned an empty response. Please try again.');
  }

  if (!parsed) {
    throw new Error('The server returned an invalid response. Please try again.');
  }

  return data;
}
