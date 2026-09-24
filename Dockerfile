# Image unique : l'API Nest sert aussi le frontend buildé (comme Django servait frontend/dist).
FROM node:22-slim AS build
RUN corepack enable
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/
RUN cd frontend && pnpm install --frozen-lockfile
COPY backend/package.json backend/pnpm-lock.yaml backend/
RUN cd backend && pnpm install --frozen-lockfile
COPY frontend frontend
COPY backend backend
RUN rm -f frontend/public/static && cd frontend && pnpm build
RUN cd backend && pnpm build && pnpm prune --prod

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/backend/dist backend/dist
COPY --from=build /app/backend/node_modules backend/node_modules
COPY --from=build /app/backend/package.json backend/
COPY --from=build /app/frontend/dist frontend/dist
COPY backend/static backend/static
# Cache des documents d'exemple déjà OCRisés (optionnel) : COPY data/document_cache data/document_cache
EXPOSE 3000
CMD ["node", "backend/dist/main.js"]
