import axios from "axios";
export const api = axios.create({ baseURL: process.env.NEXT_PUBLIC_API_URL || "" });
// TODO: wrappers getModel(), getStations(), postForecastBatch(), etc.
