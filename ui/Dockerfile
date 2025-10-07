# syntax=docker/dockerfile:1

# --- deps ---
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* .npmrc* ./
RUN npm ci || npm i

# --- build ---
FROM node:20-slim AS build
WORKDIR /app
ENV NODE_ENV=production
# If .env.production in the repo, Next will read it automatically.
# Or inject at build-time with --build-arg NEXT_PUBLIC_API_BASE=...
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# --- run ---
FROM node:20-slim AS run
WORKDIR /app
ENV NODE_ENV=production PORT=8080 HOST=0.0.0.0 NEXT_TELEMETRY_DISABLED=1
COPY --from=build /app ./
EXPOSE 8080
# IMPORTANT: no "-w"; Cloud Run expects the service to bind on $PORT
CMD ["npm", "run", "start", "--", "-p", "8080"]
