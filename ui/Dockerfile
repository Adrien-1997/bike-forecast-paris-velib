# syntax=docker/dockerfile:1
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* .npmrc* ./
RUN npm ci || npm i


FROM node:20-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build


FROM node:20-slim AS run
ENV NODE_ENV=production
WORKDIR /app
COPY --from=build /app .
ENV PORT=8080 HOST=0.0.0.0 NEXT_TELEMETRY_DISABLED=1
EXPOSE 8080
CMD ["npm", "start", "-w"]